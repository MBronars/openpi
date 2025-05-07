import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import jax.image as jimg
from typing_extensions import override
import optax

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

# from openpi.models.mask_decoder import MaskDecoder, TwoWayTransformer
# from openpi.models.prompt_encoder import PromptEncoder

from openpi.models.segmentation import PromptEncoder, MaskDecoder, TwoWayTransformer

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def _flatten_1_2(x: jnp.ndarray) -> jnp.ndarray:
    """
    Collapse dimensions 1 and 2, keep the rest unchanged.
    PyTorch's x.flatten(1, 2) analogue.
    """
    b, d1, d2, *rest = x.shape
    return x.reshape(b, d1 * d2, *rest)

def dice_loss(inputs: jnp.ndarray,
              targets: jnp.ndarray,
              *,
              eps: float = 1e-6) -> jnp.ndarray:
    """
    Soft‑Dice = 1 ‑ (2 ⟨p,t⟩ + eps)/(||p||+||t|| + eps).
    Inputs are logits; we apply sigmoid inside.
    Returns a scalar in [0, 1].
    """
    # (B, C, H, W)  or  (B, N, H, W)  →  (B*C, H*W)  or  (B*N, H*W)
    p = jax.nn.sigmoid(inputs)
    p = _flatten_1_2(p)
    t = _flatten_1_2(targets)

    num   = 2.0 * (p * t).sum(axis=-1)
    denom = p.sum(axis=-1) + t.sum(axis=-1)
    loss  = 1.0 - (num + eps) / (denom + eps)

    return loss.mean()          # scalar, <= 1.0


# ------------------------------------------------------------
# BCE-with-logits loss
# ------------------------------------------------------------
def sigmoid_ce_loss(
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """
    Binary-cross-entropy with logits, reduced the same way PyTorch does
    in the original code.
    """
    num_masks = targets.shape[0]
    # BCE-with-logits stable formulation
    loss = jnp.maximum(inputs, 0) - inputs * targets + jnp.log1p(
        jnp.exp(-jnp.abs(inputs))
    )

    loss = _flatten_1_2(loss).mean()
    
    return loss


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48
    
    # specify subgoals
    pred_segmentation: bool = True
    pred_subtask: bool = False
    pred_segmentation_only: bool = False

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        segmentation_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION], jnp.float32)
        segmentation_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                segmentations={
                    "base_0_seg": segmentation_spec,
                    "left_wrist_0_seg": segmentation_spec,
                    "right_wrist_0_seg": segmentation_spec,
                },
                segmentation_masks={
                    "base_0_seg": segmentation_mask_spec,
                    "left_wrist_0_seg": segmentation_mask_spec,
                    "right_wrist_0_seg": segmentation_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                tokenized_subtask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_subtask_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True
        elif self.pred_segmentation_only:
            filters.append(
                action_expert_params_filter,
            )

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0(_model.BaseModel):
    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        self.pred_segmentation = config.pred_segmentation
        self.pred_subtask = config.pred_subtask
        self.max_token_len = config.max_token_len
        if config.pred_segmentation:
            self.seg_tokens = nnx.Param(jax.random.normal(rngs.params(), (1, 3, paligemma_config.width)))
            self.mask_decoder = MaskDecoder(
                transformer_dim=paligemma_config.width,
                transformer=TwoWayTransformer(
                    depth=2, embedding_dim=paligemma_config.width, mlp_dim=2048, num_heads=8, rngs=rngs
                ),
                num_multimask_outputs=1,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                rngs=rngs,
            )
            self.prompt_encoder = PromptEncoder(
                embed_dim=paligemma_config.width,
                image_embedding_size=(16, 16),
                input_image_size=(224, 224),
                mask_in_chans=16,
                rngs=rngs,
            )
            

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask
    
    def embed_subgoals(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        
        batch_shape = obs.images["base_0_rgb"].shape[0]
        
        # add subtask language
        if obs.tokenized_subtask is not None and self.pred_subtask:
            tokenized_subtask = self.PaliGemma.llm(obs.tokenized_subtask, method="embed")
            tokens.append(tokenized_subtask)
            input_mask.append(obs.tokenized_subtask_mask)
            # full attention between image and language inputs
            ar_mask += [True] * tokenized_subtask.shape[1]
        
        # embed segmentation tokens
        for i, name in enumerate(obs.segmentations):
            
            seg_token = jnp.repeat(self.seg_tokens[:, i : i+1, :], repeats=batch_shape, axis=0)

            tokens.append(seg_token)
            input_mask.append(
                einops.repeat(
                    obs.segmentation_masks[name],
                    "b -> b s",
                    s=1,
                )
            )
            # image tokens attend to each other, but not to the language
            if i == 0:
                ar_mask += [True]
            else:
                ar_mask += [False]
            
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask
            
        

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # add a single state token
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # image/language inputs do not attend to state or actions
        ar_mask += [True]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # mix timestep + action information using an MLP
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask
    
    def generate_segmentation_masks(
        self,
        seg_out: jnp.ndarray,        # (B, N_seg, C)  – whatever your prompt encoder expects
        img_embs: jnp.ndarray,       # (B * num_cam, H*W, C) or any shape you currently feed in
    ) -> jnp.ndarray:
        """
        Returns masks resized to `self.config.resize_imgs_with_padding`
        """
        # -------------------------------------------------------------
        # 1. reshape the image embeddings exactly like the Torch code
        # -------------------------------------------------------------
        batch_size = img_embs.shape[0]                        # infer B
        hidden_size = img_embs.shape[-1]                     # infer C
        h, w, c = 16, 16, hidden_size

        img_embs = img_embs.reshape(batch_size, 3, h * w, c)
        img_embs = img_embs.reshape(batch_size * 3, h, w, c)
        img_embs = jnp.transpose(img_embs, (0, 3, 1, 2))

        # -------------------------------------------------------------
        # 2. prompt encoder (returns JAX arrays already on device)
        # -------------------------------------------------------------
        sparse_emb, dense_emb = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=seg_out,
        )
        
        # -------------------------------------------------------------
        # 3. decode masks
        # -------------------------------------------------------------
        pe = self.prompt_encoder.get_dense_pe()  # Get the PE once
        pe_shape = pe.shape[1:]  # Extract the shape statically
        image_pe = jnp.broadcast_to(pe, (img_embs.shape[0],) + pe_shape)
        
        low_res_masks, iou_pred = self.mask_decoder(
            image_embeddings           = img_embs,
            image_pe                   = image_pe,
            sparse_prompt_embeddings   = sparse_emb,
            dense_prompt_embeddings    = dense_emb,
            multimask_output           = False,
        )
        
        # -------------------------------------------------------------
        # 4. upsample to full resolution (bilinear, align_corners=False)
        # -------------------------------------------------------------
        out_h, out_w = (224, 224)   # Tuple[int, int]
        # low_res_masks: (B, 1, h_m, w_m) – keep (B, 1) dims, resize H/W only
        full_masks = jimg.resize(
            low_res_masks.astype(jnp.float32),
            shape = (low_res_masks.shape[0], low_res_masks.shape[1], out_h, out_w),
            method = "linear",
            antialias = False,
        )

        return full_masks
        
        

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)

        observation = _model.preprocess_observation(preprocess_rng, observation, train=train, segmentation=self.pred_segmentation)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        subgoal_tokens, subgoal_mask, subgoal_ar_mask = self.embed_subgoals(observation)
        
        orig_prefix_len = prefix_tokens.shape[1]
        prefix_tokens = jnp.concatenate([prefix_tokens, subgoal_tokens], axis=1)
        prefix_mask = jnp.concatenate([prefix_mask, subgoal_mask], axis=1)
        prefix_ar_mask = jnp.concatenate([prefix_ar_mask, subgoal_ar_mask], axis=0)
        
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )
        
        subgoal_out = prefix_out[:, orig_prefix_len:]
        prefix_out = prefix_out[:, :orig_prefix_len]

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        
        # total_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        
        total_loss = 0
        
        if self.pred_subtask:
            subtask_out = subgoal_out[:, :-3]
            subtask_logits = self.PaliGemma.llm(subtask_out, method="decode")
            
            subtask_logits = subtask_logits[:, :-1]
            
            gt_tokens = observation.tokenized_subtask[:, 1:]
            gt_mask = observation.tokenized_subtask_mask[:, 1:]
            
            # compute the cross entropy loss
            ce = optax.softmax_cross_entropy_with_integer_labels(
                logits=subtask_logits,
                labels=gt_tokens
            )
            
            loss = jnp.sum(ce * gt_mask) / jnp.sum(gt_mask)
            #total_loss += 0.1 * loss
            
        if self.pred_segmentation:
            img_tokens = prefix_tokens[:, :(orig_prefix_len-self.max_token_len)]
            seg_out = subgoal_out[:, -3:]
            seg_out = seg_out.reshape(-1, 1, seg_out.shape[-1])
        
            pred_seg = self.generate_segmentation_masks(seg_out, img_tokens)
            
            gt_segs = []
            for i, name in enumerate(observation.segmentations):
                gt_seg = observation.segmentations[name]
                gt_seg = gt_seg.reshape(-1, 1, 224, 224)
                gt_segs.append(gt_seg)
            gt_segs = jnp.concatenate(gt_segs, axis=0)
                    
            seg_dice_loss = dice_loss(pred_seg, gt_segs)
            seg_bce_loss = sigmoid_ce_loss(pred_seg, gt_segs)
     
            total_loss += 0.5 * seg_bce_loss + 0.5 * seg_dice_loss
        
        return total_loss

    def sample_actions_with_subgoals(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ):
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        
        original_prefix_len = prefix_tokens.shape[1]
        
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_out, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        
        language_ar_mask = jnp.ones((self.max_token_len), dtype=jnp.bool)  # Current token is always valid
        language_mask = jnp.ones((batch_size, self.max_token_len), dtype=jnp.bool)  # Current token is always valid
        
        is_eos = jnp.zeros((batch_size,), dtype=jnp.bool)  # Current token is always valid
        
        def language_step(carry):
            curr_language, curr_idx, is_eos = carry
            
            eos_id = 1
            pad_id = 0
            first_eos_idx = jnp.argmax(curr_language == eos_id, axis=1)  
            has_eos = jnp.any(curr_language == eos_id, axis=1)               

            positions = jnp.arange(curr_language.shape[1])[None, :]
            after_eos = (positions > first_eos_idx[:, None]) & has_eos[:, None]
            
            curr_language = jnp.where(after_eos, pad_id, curr_language)
            curr_language_mask = (curr_language != 0) 
            step_mask = jnp.arange(self.max_token_len) <= curr_idx  
            curr_language_mask = curr_language_mask & step_mask
            
            curr_language_ar_mask = language_ar_mask
            
            full_mask = jnp.concatenate([prefix_mask, curr_language_mask], axis=-1)
            full_ar_mask = jnp.concatenate([prefix_ar_mask, language_ar_mask], axis=-1)
            full_attn_mask = make_attn_mask(full_mask, full_ar_mask)
            language_attn_mask = full_attn_mask[:, -self.max_token_len:, :]
                        
            # Add language token positional info
            language_positions = jnp.cumsum(full_mask, axis=1) - 1
            language_positions = language_positions[:, -self.max_token_len:]
            
            curr_lang_embed = self.PaliGemma.llm(curr_language, method="embed")
            (subtask_out, _), _ = self.PaliGemma.llm(
                [curr_lang_embed, None], 
                mask=language_attn_mask, 
                positions=language_positions, 
                kv_cache=kv_cache
            )
            
            # Get logits for next token prediction
            subtask_logits = self.PaliGemma.llm(subtask_out, method="decode")
            next_tokens = jnp.argmax(subtask_logits, axis=-1) 
            curr_token = next_tokens[:, curr_idx]
        
            is_next_eos = (curr_token == eos_id)  
            is_eos = jnp.logical_or(is_eos, jnp.any(is_next_eos, axis=-1))

            curr_language = curr_language.at[:, curr_idx+1].set(curr_token)
            
            return (curr_language, curr_idx + 1, is_eos)

        def language_cond(carry):
            curr_language, curr_idx, is_eos = carry
            
            # Continue until we've generated the desired length or hit an end token
            is_not_done_length = curr_idx < self.max_token_len - 1
            
            all_eos = jnp.all(is_eos)
            is_not_done = jnp.logical_and(is_not_done_length, ~all_eos)
            
            return is_not_done
        

        sep_tokens = jnp.full((batch_size, 1), 108, dtype=jnp.int32)  # Assuming 108 is the ID for the separator token
        pad_tokens = jnp.full((batch_size, self.max_token_len-1), 0, dtype=jnp.int32)  # Assuming 108 is the ID for the separator token
        language_tokens = jnp.concatenate([sep_tokens, pad_tokens], axis=1)
        
        pred_subtask, _, final_eos = jax.lax.while_loop(language_cond, language_step, (language_tokens, jnp.array(0), is_eos))
        
        pred_subtask_mask = (pred_subtask != 0) 
        full_mask = jnp.concatenate([prefix_mask, pred_subtask_mask], axis=-1)
        full_ar_mask = jnp.concatenate([prefix_ar_mask, language_ar_mask], axis=-1)
        full_attn_mask = make_attn_mask(full_mask, full_ar_mask)
        language_attn_mask = full_attn_mask[:, -self.max_token_len:, :]
                    
        # Add language token positional info
        language_positions = jnp.cumsum(full_mask, axis=1) - 1
        language_positions = language_positions[:, -self.max_token_len:]
        
        curr_lang_embed = self.PaliGemma.llm(pred_subtask, method="embed")
        
        _, final_kv_cache = self.PaliGemma.llm(
                [curr_lang_embed, None], 
                mask=language_attn_mask, 
                positions=language_positions, 
                kv_cache=kv_cache
        )
        

        segmentation_ar_mask = [1] + [0] + [0]
        segmentation_ar_mask = jnp.array(segmentation_ar_mask, dtype=jnp.bool_)
        
        segmentation_mask = []
        for name, val in observation.image_masks.items():
            curr_segmentation_mask = einops.repeat(
                val,
                "b -> b s",
                s=1,
            )
            segmentation_mask.append(curr_segmentation_mask)
        segmentation_mask = jnp.concatenate(segmentation_mask, axis=1)
        
        updated_lang_mask = pred_subtask != 0
        
        prefix_mask = jnp.concatenate([prefix_mask, updated_lang_mask, segmentation_mask], axis=1)
        full_segmentation_ar_mask = jnp.concatenate([prefix_ar_mask, language_ar_mask, segmentation_ar_mask], axis=0)
        
        segmentation_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        
        full_segmentation_attn_mask = make_attn_mask(prefix_mask, full_segmentation_ar_mask)
        full_segmentation_attn_mask = full_segmentation_attn_mask[:, -3:, :]
        segmentation_positions = segmentation_positions[:, -3:]
        
        seg_tokens = jnp.repeat(self.seg_tokens, repeats=batch_size, axis=0)
        
        
        (subtask_out, _), final_kv_cache = self.PaliGemma.llm(
                [seg_tokens, None], 
                mask=full_segmentation_attn_mask, 
                positions=segmentation_positions, 
                kv_cache=final_kv_cache
        )
        
        img_tokens = prefix_tokens[:, :(original_prefix_len-self.max_token_len)]
        
        seg_out = subtask_out[:, -3:]
        seg_out = seg_out.reshape(-1, 1, seg_out.shape[-1])
        
        full_masks = self.generate_segmentation_masks(seg_out, img_tokens)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + self.max_token_len + 3 + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=final_kv_cache
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2


        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        full_masks = jnp.transpose(full_masks, (1, 0, 2, 3))
        return x_0, (pred_subtask, full_masks)
    
    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_out, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        
        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0