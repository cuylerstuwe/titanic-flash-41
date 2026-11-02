"""Hoist immutable routing-map loads out of the grouped FC1 K loop.

Adapted from b12x W4A8MaterializedPhase1Kernel (Apache-2.0), pinned commit
a7d7d29b2ef8869086e0ceaa787321f17544e3c9.
https://github.com/local-inference-lab/b12x

The completed routing producer owns token_map. Within a published expert task,
the activation/scale token for each loader thread is independent of K. Keep
those two integer row offsets in registers through all80 K64 slices. Loads,
shared destinations, padding token0, barriers, MMA, activation and output
quantization remain unchanged. Scope is M16/N128, native packed N64 tail.
"""
import ast


def patch(source):
    original=ast.parse(source)
    guard='        self.tile_m = 16 if source_tile_m == 16 else 64\n'
    assert source.count(guard)==1
    source=source.replace(guard,
        '        assert source_tile_m == 16 and n64_repacked and n64_tail and trellis_bits is None\n'+guard)
    start=source.index('    def _stage_slice(');end=source.index('    @cute.jit',start+10)
    stage=source[start:end]
    anchor='        packed_w13_tiles: Int32,\n'
    assert stage.count(anchor)==1
    stage=stage.replace(anchor,anchor+'        a_row_word_base: Int64,\n        sf_row_byte_base: Int64,\n')
    begin=stage.index('        physical_row_base = Int64(source_m_tile)')
    finish=stage.index('        k128_slice = k64_slice >> Int32(1)',begin)
    old=stage[begin:finish]
    assert old.count('token_map[')==2 and old.count('cp_async4_shared_global(')==1
    stage=stage[:begin]+'''        # Routing is fixed for this task; row offsets were loaded once.
        if tid < Int32(self.tile_m * 4):
            row = tid >> Int32(2)
            vec = tid & Int32(3)
            physical_vec = vec ^ (row & Int32(7))
            src_word = a_row_word_base + Int64(k64_slice * 16 + vec * 4)
            cp_async4_shared_global(
                a_base + row * Int32(128) + (physical_vec << Int32(4)),
                get_ptr_as_int64(packed_a_u32, src_word),
            )
        if tid < Int32(self.tile_m):
            sf_src = sf_row_byte_base + Int64(k64_slice >> 1) * Int64(4)
            cp_async_u32_shared_global(
                sfa_base + (tid << Int32(2)),
                get_ptr_as_int64(scale_storage, sf_src),
            )

'''+stage[finish:]
    source=source[:start]+stage+source[end:]
    start=source.index('    def _run_task(');end=source.index('    @cute.kernel',start)
    run=source[start:end]
    anchor='        input_k64_tiles = input_k128_tiles * Int32(2)\n'
    assert run.count(anchor)==1
    run=run.replace(anchor,'''        physical_row_base = Int64(source_m_tile) * Int64(self.source_tile_m) + Int64(m_half * self.tile_m)
        a_row_word_base = Int64(0)
        sf_row_byte_base = Int64(0)
        if tid < Int32(self.tile_m * 4):
            row = tid >> Int32(2)
            tok = Int32(0)
            if row < valid_rows:
                tok = token_map[physical_row_base + row].to(Int32)
                if cutlass.const_expr(self.deterministic_output):
                    tok = tok // Int32(self.num_topk)
            a_row_word_base = Int64(tok) * Int64(input_k128_tiles * Int32(32))
        if tid < Int32(self.tile_m):
            tok = Int32(0)
            if tid < valid_rows:
                tok = token_map[physical_row_base + tid].to(Int32)
                if cutlass.const_expr(self.deterministic_output):
                    tok = tok // Int32(self.num_topk)
            sf_row_byte_base = Int64(tok) * Int64(input_k128_tiles) * Int64(4)

'''+anchor)
    anchor='                    packed_w13_tiles,\n                )'
    assert run.count(anchor)==2
    run=run.replace(anchor,'                    packed_w13_tiles,\n                    a_row_word_base,\n                    sf_row_byte_base,\n                )')
    source=source[:start]+run+source[end:]
    result=ast.parse(source)
    assert [n.name for n in ast.walk(result)if isinstance(n,ast.FunctionDef)]==[
        n.name for n in ast.walk(original)if isinstance(n,ast.FunctionDef)]
    return source
