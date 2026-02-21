import time
import pandas as pd
import jax
import jax.numpy as jnp
import warnings

warnings.filterwarnings('ignore')

import sys
import os
# Ensure pgx is imported from the local directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'pgx')))

import pgx
from pgx.experimental.wrappers import auto_reset

def act_randomly(rng, legal_action_mask):
    """Uniformly sample an index that is legal in every row of the batch."""
    logits = jnp.log(legal_action_mask.astype(jnp.float32))       # 1 → 0, 0 → ‑inf
    return jax.random.categorical(rng, logits=logits, axis=1)     # shape = (B,)

def make_env_fns(B: int):
    env = pgx.make("war_chest_simplified")
    
    env_init     = jax.jit(jax.vmap(env.init))
    env_step     = jax.jit(jax.vmap(auto_reset(env.step, env.init)))
    iterations   = 2048

    act_rand = jax.jit(act_randomly)

    def loop(state):
        def body(carry, _):
            rng, s = carry
            rng, sub1, sub2 = jax.random.split(rng, 3)
            
            # Action sampling
            a = act_rand(sub1, s.legal_action_mask)
            
            # Env step requires batched keys for War Chest Simplified
            keys = jax.random.split(sub2, B)
            s = env_step(s, a, keys)
            
            return (rng, s), None

        rng0 = jax.random.PRNGKey(0)
        (rng_f, s_f), _ = jax.lax.scan(body,
                                       (rng0, state),
                                       None,
                                       length=iterations)
        return s_f

    return env_init, loop, iterations

if __name__ == "__main__":
    print("Devices:", jax.devices())
    
    batch_sizes = [512, 1024, 2048, 4096, 8192]
    records     = []

    for B in batch_sizes:
        env_init, loop, iters = make_env_fns(B)

        # (re‑)build initial state
        keys   = jax.random.split(jax.random.PRNGKey(42), B)
        state  = env_init(keys)

        # compile once
        compiled = jax.jit(loop).lower(state).compile()

        # warm‑up run
        compiled(state).terminated.block_until_ready()

        # timed run
        t0 = time.perf_counter()
        compiled(state).terminated.block_until_ready()
        wall = time.perf_counter() - t0

        steps   = iters * B          # total env‑steps executed on device
        records.append(dict(Batch=B,
                            Wall_s=round(wall, 4),
                            Steps_per_s=int(steps / wall)))

    df = pd.DataFrame(records)
    print(df.to_markdown(index=False))