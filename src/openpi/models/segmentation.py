from __future__ import annotations

from typing import Callable, List, Optional, Tuple
import math
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# -----------------------------------------------------------------------------
# Basic building blocks
# -----------------------------------------------------------------------------

class MLPBlock(nnx.Module):
    """Dense → activation → Dense."""

    def __init__(self, embedding_dim: int, mlp_dim: int, *, act: Callable = jax.nn.gelu, rngs: nnx.Rngs):
        super().__init__()
        self.lin1 = nnx.Linear(embedding_dim, mlp_dim, rngs=rngs)
        self.lin2 = nnx.Linear(mlp_dim, embedding_dim, rngs=rngs)
        self.act = act

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nnx.Module):
    """Channel‑wise LayerNorm for tensors shaped (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        super().__init__()
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((num_channels,), dtype=jnp.float32))
        self.bias = nnx.Param(jnp.zeros((num_channels,), dtype=jnp.float32))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        u = jnp.mean(x, axis=-1, keepdims=True)
        s = jnp.mean(jnp.square(x - u), axis=-1, keepdims=True)
        x = (x - u) / jnp.sqrt(s + self.eps)
        return x * self.weight[None, None, None, :] + self.bias[None, None, None, :]


class MLP(nnx.Module):
    """(Dense + ReLU)×(n‑1) → Dense, optional final sigmoid."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, *, sigmoid_output: bool = False, rngs: nnx.Rngs):
        super().__init__()
        # Change: Use string keys instead of integer indices
        self.layers = nnx.Dict({})
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # Use f-string to ensure string keys
            self.layers[f'layer_{i}'] = nnx.Linear(in_dim, out_dim, rngs=rngs)
        self.sigmoid_output = sigmoid_output

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(self.layers.values()):
            x = jax.nn.relu(layer(x)) if i < len(self.layers.values()) - 1 else layer(x)
        return jax.nn.sigmoid(x) if self.sigmoid_output else x


class Attention(nnx.Module):
    """Multi‑head scaled dot‑product attention (eager).
    Q, K, V have shape (B, N, C).
    """

    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1, *, rngs: nnx.Rngs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.internal_dim = embedding_dim // downsample_rate
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim"

        self.q_proj = nnx.Linear(embedding_dim, self.internal_dim, rngs=rngs)
        self.k_proj = nnx.Linear(embedding_dim, self.internal_dim, rngs=rngs)
        self.v_proj = nnx.Linear(embedding_dim, self.internal_dim, rngs=rngs)
        self.out_proj = nnx.Linear(self.internal_dim, embedding_dim, rngs=rngs)

    # helpers -----------------------------------------------------------------
    def _split_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        b, n, c = x.shape
        x = x.reshape(b, n, self.num_heads, c // self.num_heads)
        return jnp.transpose(x, (0, 2, 1, 3))  # (B, H, N, C//H)

    def _merge_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        b, h, n, d = x.shape
        x = jnp.transpose(x, (0, 2, 1, 3))  # (B, N, H, D)
        return x.reshape(b, n, h * d)

    # forward -----------------------------------------------------------------
    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        q, k, v = map(self._split_heads, (q, k, v))

        scale = 1.0 / math.sqrt(q.shape[-1])
        attn = jnp.einsum("bhqd,bhkd->bhqk", q * scale, k)
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.einsum("bhqk,bhvd->bhqd", attn, v)
        out = self._merge_heads(out)
        return self.out_proj(out)


# -----------------------------------------------------------------------------
# Two‑way transformer (image ↔ token)
# -----------------------------------------------------------------------------

class TwoWayAttentionBlock(nnx.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        *,
        activation: Callable = jax.nn.relu,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.skip_first_layer_pe = skip_first_layer_pe
        self.activation = activation

        self.self_attn = Attention(embedding_dim, num_heads, rngs=rngs)
        self.norm1 = nnx.LayerNorm(embedding_dim, rngs=rngs)

        self.cross_token_to_img = Attention(embedding_dim, num_heads, attention_downsample_rate, rngs=rngs)
        self.norm2 = nnx.LayerNorm(embedding_dim, rngs=rngs)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, act=activation, rngs=rngs)
        self.norm3 = nnx.LayerNorm(embedding_dim, rngs=rngs)

        self.cross_img_to_token = Attention(embedding_dim, num_heads, attention_downsample_rate, rngs=rngs)
        self.norm4 = nnx.LayerNorm(embedding_dim, rngs=rngs)

    # ------------------------------------------------------------------ forward
    def __call__(
        self,
        queries: jnp.ndarray,
        keys: jnp.ndarray,
        query_pe: jnp.ndarray,
        key_pe: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:  # type: ignore[override]
        # --- self‑attention over queries ------------------------------------
        if self.skip_first_layer_pe:
            attn_out = self.self_attn(queries, queries, queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q, q, queries)
        queries = self.norm1(queries + attn_out)

        # --- cross‑attention: tokens → image --------------------------------
        q, k = queries + query_pe, keys + key_pe
        attn_out = self.cross_token_to_img(q, k, keys)
        queries = self.norm2(queries + attn_out)

        # --- MLP ------------------------------------------------------------
        queries = self.norm3(queries + self.mlp(queries))

        # --- cross‑attention: image → tokens --------------------------------
        q, k = queries + query_pe, keys + key_pe
        attn_out = self.cross_img_to_token(k, q, queries)
        keys = self.norm4(keys + attn_out)

        return queries, keys


class TwoWayTransformer(nnx.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        *,
        activation: Callable = jax.nn.relu,
        attention_downsample_rate: int = 2,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        # Change: Use string block_i instead of integer indices
        self.layers = nnx.Dict({
            f"block_{i}": TwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                activation=activation,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=(i == 0),
                rngs=rngs,
            )
            for i in range(depth)
        })
        self.final_attn = Attention(embedding_dim, num_heads, attention_downsample_rate, rngs=rngs)
        self.norm_final = nnx.LayerNorm(embedding_dim, rngs=rngs)

    # ------------------------------------------------------------------ forward
    def __call__(
        self,
        image_embedding: jnp.ndarray,  # (B, C, H, W)
        image_pe: jnp.ndarray,        # (B, C, H, W)
        point_embedding: jnp.ndarray, # (B, Nq, C)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:  # type: ignore[override]
        b, c, h, w = image_embedding.shape
        tokens_img = jnp.transpose(image_embedding.reshape(b, c, h * w), (0, 2, 1))
        pe_img = jnp.transpose(image_pe.reshape(b, c, h * w), (0, 2, 1))
        
        queries, keys = point_embedding, tokens_img
        for layer in self.layers.values():
            queries, keys = layer(queries, keys, point_embedding, pe_img)

        attn_out = self.final_attn(queries + point_embedding, keys + pe_img, keys)
        queries = self.norm_final(queries + attn_out)
        return queries, keys


# -----------------------------------------------------------------------------
# Mask decoder (image → masks)
# -----------------------------------------------------------------------------

class MaskDecoder(nnx.Module):
    def __init__(
        self,
        transformer_dim: int,
        transformer: TwoWayTransformer,
        *,
        num_multimask_outputs: int = 3,
        activation: Callable = jax.nn.gelu,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.transformer = transformer
        self.activation = activation
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1

        # learned tokens -----------------------------------------------------
        self.iou_token = nnx.Embed(num_embeddings=1, features=transformer_dim, rngs=rngs)
        self.mask_tokens = nnx.Embed(num_embeddings=self.num_mask_tokens, features=transformer_dim, rngs=rngs)

        # upscaling conv stack ----------------------------------------------
        self.conv_up_1 = nnx.ConvTranspose(transformer_dim, transformer_dim // 4, kernel_size=(2, 2), strides=(2, 2), rngs=rngs)
        self.ln_up_1 = LayerNorm2d(transformer_dim // 4, rngs=rngs)
        self.conv_up_2 = nnx.ConvTranspose(transformer_dim // 4, transformer_dim // 8, kernel_size=(2, 2), strides=(2, 2), rngs=rngs)

        # mlps ---------------------------------------------------------------
        # Change: Use string keys for output_hypernets
        self.output_hypernets = nnx.Dict({
            f"mlp_{i}": MLP(transformer_dim, transformer_dim, transformer_dim // 8, num_layers=3, rngs=rngs)
            for i in range(self.num_mask_tokens)
        })
        self.iou_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth, rngs=rngs)

    # ------------------------------------------------------------------ utils
    def _output_upscale(self, src: jnp.ndarray) -> jnp.ndarray:
        x = self.conv_up_1(src)
        x = self.activation(self.ln_up_1(x))
        x = self.activation(self.conv_up_2(x))
        return x

    # ------------------------------------------------------------------ api
    def __call__(
        self,
        image_embeddings: jnp.ndarray,      # (B, C, H, W)
        image_pe: jnp.ndarray,              # (B, C, H, W)
        sparse_prompt_embeddings: jnp.ndarray,  # (B, Nt, C)
        dense_prompt_embeddings: jnp.ndarray,   # (B, C, H, W)
        *,
        multimask_output: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:  # type: ignore[override]
        masks, iou_pred = self.predict_masks(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
        )
        slice_ = slice(1, None) if multimask_output else slice(0, 1)
        return masks[:, slice_, :, :], iou_pred[:, slice_]

    # ------------------------------------------------------------------ core
    def predict_masks(
        self,
        image_embeddings: jnp.ndarray,
        image_pe: jnp.ndarray,
        sparse_prompt_embeddings: jnp.ndarray,
        dense_prompt_embeddings: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        b = image_embeddings.shape[0]
        iou_tok = self.iou_token(jnp.zeros((b, 1), dtype=jnp.int32))
        mask_tok_indices = jnp.broadcast_to(jnp.arange(self.num_mask_tokens, dtype=jnp.int32), (b, self.num_mask_tokens))
        mask_toks = self.mask_tokens(mask_tok_indices)
        tokens = jnp.concatenate([iou_tok, mask_toks, sparse_prompt_embeddings], axis=1)

        src = image_embeddings + dense_prompt_embeddings
        b, c, h, w = src.shape
        hs, src_flat = self.transformer(src, image_pe, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : 1 + self.num_mask_tokens, :]

        # upscale ----------------------------------------------------------------
        # 1️⃣  (B, C, H, W) ➔ (B, H, W, C) for ConvTranspose
        src_nhwc = src_flat.reshape(b, h, w, c)

        # 2️⃣  run the up‑scaling stack, then bring channels back to axis 1
        src_up = jnp.transpose(self._output_upscale(src_nhwc), (0, 3, 1, 2))  # (B,C,H↑,W↑)
        b, c, h, w = src_up.shape
        src_up_flat = src_up.reshape(b, c, h * w)
        
        # hypernet ---------------------------------------------------------------
        # Use enumerate with string keys
        hypernet_outputs = []
        for i in range(self.num_mask_tokens):
            mlp = self.output_hypernets[f"mlp_{i}"]
            hypernet_outputs.append(mlp(mask_tokens_out[:, i, :]))
        hyper_in = jnp.stack(hypernet_outputs, axis=1)
        masks = jnp.einsum("bmc,bcn->bmn", hyper_in, src_up_flat)  # (b, m, h*w)
        masks = masks.reshape(b, self.num_mask_tokens, h, w)       # (b, m, h, w)
        # masks = jnp.einsum("bmc,bch->bmhw", hyper_in, src_up_flat).reshape(b, self.num_mask_tokens, h, w)

        iou_pred = self.iou_head(iou_token_out)
        return masks, iou_pred


# -----------------------------------------------------------------------------
# Prompt encoder
# -----------------------------------------------------------------------------

class PositionEmbeddingRandom(nnx.Module):
    def __init__(self, num_pos_feats: int = 64, *, scale: Optional[float] = None, rngs: nnx.Rngs):
        super().__init__()
        self.scale = 1.0 if not scale or scale <= 0.0 else scale
        key = rngs.params()                
        
        # array = self.scale * jax.random.normal(key, (2, num_pos_feats), dtype=jnp.float32)
        # self.gauss = nnx.Buffer(jax.lax.stop_gradient(array))   
        self.gauss = nnx.Param(               # stored, device‑moved, but not optimised
            jax.lax.stop_gradient(
                self.scale * jax.random.normal(key, (2, num_pos_feats), dtype=jnp.float32)
            )
        )

    # helpers ------------------------------------------------------------------
    def _pe(self, coords: jnp.ndarray) -> jnp.ndarray:
        coords = (coords * 2.0) - 1.0
        # Explicit einsum rather than matmul
        gauss  = self.gauss.value  
        coords = jnp.einsum('bij,jk->bik', coords, gauss)
        coords = 2 * jnp.pi * coords
        return jnp.concatenate([jnp.sin(coords), jnp.cos(coords)], axis=-1)

    # ------------------------------------------------------------------ api
    def __call__(self, size: Tuple[int, int]) -> jnp.ndarray:  # type: ignore[override]
        h, w = size
        y, x = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
        coords = jnp.stack([(x + 0.5) / w, (y + 0.5) / h], axis=-1)
        pe = self._pe(coords)
        return jnp.transpose(pe, (2, 0, 1))

    def forward_with_coords(self, coords: jnp.ndarray, image_size: Tuple[int, int]) -> jnp.ndarray:
        coords_norm = coords.copy()
        coords_norm = coords_norm.at[:, :, 0].set(coords_norm[:, :, 0] / image_size[1])
        coords_norm = coords_norm.at[:, :, 1].set(coords_norm[:, :, 1] / image_size[0])
        return self._pe(coords_norm)


class PromptEncoder(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        *,
        activation: Callable = jax.nn.gelu,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.activation = activation

        # positional encoding -------------------------------------------------
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2, rngs=rngs)

        # point & box embeddings ---------------------------------------------
        self.num_point_embeddings = 4  # pos, neg, +2 box corners
        # Change: Use explicit string names instead of indexed list for point embeddings
        self.point_embeddings = nnx.Dict({
            f"point_emb_{i}": nnx.Embed(1, embed_dim, rngs=rngs) 
            for i in range(self.num_point_embeddings)
        })
        self.not_a_point_embed = nnx.Embed(1, embed_dim, rngs=rngs)

        # mask downscaling ----------------------------------------------------
        ch_mid = mask_in_chans // 4
        self.mask_conv1 = nnx.Conv(1, ch_mid, kernel_size=(2, 2), strides=(2, 2), rngs=rngs)
        self.mask_ln1 = LayerNorm2d(ch_mid, rngs=rngs)
        self.mask_conv2 = nnx.Conv(ch_mid, mask_in_chans, kernel_size=(2, 2), strides=(2, 2), rngs=rngs)
        self.mask_ln2 = LayerNorm2d(mask_in_chans, rngs=rngs)
        self.mask_conv3 = nnx.Conv(mask_in_chans, embed_dim, kernel_size=(1, 1), rngs=rngs)
        self.no_mask_embed = nnx.Embed(1, embed_dim, rngs=rngs)
        
    def get_dense_pe(self) -> jnp.ndarray:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.
        """
        # self.image_embedding_size
        out = self.pe_layer(self.image_embedding_size)
        return jnp.expand_dims(out, axis=0)  # (1, embed_dim, H, W)
    
    # def get_dense_pe(self) -> jnp.ndarray:
    #     """
    #     Returns the positional encoding used to encode point prompts,
    #     applied to a dense set of points the shape of the image encoding.
    #     """
    #     # Define a pure function that doesn't depend on self
    #     def _get_pe(size):
    #         # Call pe_layer directly
    #         h, w = size
    #         y, x = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    #         coords = jnp.stack([(x + 0.5) / w, (y + 0.5) / h], axis=-1)
            
    #         # Apply the positional encoding directly
    #         coords = (coords * 2.0) - 1.0
    #         coords = jnp.einsum('bij,jk->bik', coords, self.pe_layer.gauss)
    #         coords = 2 * jnp.pi * coords
    #         pe = jnp.concatenate([jnp.sin(coords), jnp.cos(coords)], axis=-1)
    #         return jnp.transpose(pe, (2, 0, 1))
        
    #     # Use the pure function with concrete values
    #     out = _get_pe(self.image_embedding_size)
        
    #     # Apply the expand_dims operation
    #     return jnp.expand_dims(out, axis=0)  # (1, embed_dim, H, W)

    # ------------------------------------------------------------------ helpers
    def _embed_points(self, points: jnp.ndarray, labels: jnp.ndarray, pad: bool) -> jnp.ndarray:
        points = points + 0.5
        if pad:
            b = points.shape[0]
            points = jnp.concatenate([points, jnp.zeros((b, 1, 2))], axis=1)
            labels = jnp.concatenate([labels, -jnp.ones((b, 1))], axis=1)

        pe = self.pe_layer.forward_with_coords(points, self.input_image_size)
        nap_mask = (labels == -1)[:, :, None]
        pos_mask = (labels == 0)[:, :, None]
        neg_mask = (labels == 1)[:, :, None]

        emb_nap = self.not_a_point_embed(jnp.zeros((1,), dtype=jnp.int32))
        # Get embeddings by string keys
        emb_pos = self.point_embeddings["point_emb_0"](jnp.zeros((1,), dtype=jnp.int32))
        emb_neg = self.point_embeddings["point_emb_1"](jnp.zeros((1,), dtype=jnp.int32))

        out = pe * (~nap_mask)
        out = out + nap_mask * emb_nap + pos_mask * emb_pos + neg_mask * emb_neg
        return out

    def _embed_boxes(self, boxes: jnp.ndarray) -> jnp.ndarray:
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        pe = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        # Get embeddings by string keys
        emb0 = self.point_embeddings["point_emb_2"](jnp.zeros((1,), dtype=jnp.int32))
        emb1 = self.point_embeddings["point_emb_3"](jnp.zeros((1,), dtype=jnp.int32))
        pe = pe.at[:, 0, :].add(emb0).at[:, 1, :].add(emb1)
        return pe

    def _embed_masks(self, masks: jnp.ndarray) -> jnp.ndarray:
        x = self.activation(self.mask_ln1(self.mask_conv1(masks)))
        x = self.activation(self.mask_ln2(self.mask_conv2(x)))
        return self.mask_conv3(x)

    def _batch_size(self, pts, boxes, masks, texts):
        for item in (pts, boxes, masks, texts):
            if item is not None:
                return item[0].shape[0] if isinstance(item, tuple) else item.shape[0]
        return 1

    # ------------------------------------------------------------------ forward
    def __call__(
        self,
        *,
        points: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        boxes: Optional[jnp.ndarray] = None,
        masks: Optional[jnp.ndarray] = None,
        text_embeds: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:  # type: ignore[override]
        bs = self._batch_size(points, boxes, masks, text_embeds)
        sparse = jnp.zeros((bs, 0, self.embed_dim))

        if points is not None:
            pts, labels = points
            sparse = jnp.concatenate([sparse, self._embed_points(pts, labels, pad=(boxes is None))], axis=1)
        if boxes is not None:
            sparse = jnp.concatenate([sparse, self._embed_boxes(boxes)], axis=1)
        if text_embeds is not None:
            sparse = jnp.concatenate([sparse, text_embeds], axis=1)

        if masks is not None:
            dense = self._embed_masks(masks)
        else:
            no_mask = self.no_mask_embed(jnp.zeros((1,), dtype=jnp.int32))
            dense = jnp.broadcast_to(no_mask.reshape(1, -1, 1, 1), (bs, self.embed_dim, *self.image_embedding_size))
        return sparse, dense