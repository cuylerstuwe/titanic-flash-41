@triton.jit
def select_image(X,O,T,H:tl.constexpr,B:tl.constexpr):
    program=tl.program_id(0)
    row=program//20
    col=(program%20)*B+tl.arange(0,B)
    if tl.load(T+row)==129264:
        raw=tl.load(X+row*H+col,col<H,0)
        tl.store(O+row*H+col,raw,col<H)
