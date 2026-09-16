# sets/demo.djs -- run with: python app.py --set sets/demo.djs
# Transport does not auto-start; type start() once the UI is up.

bpm(128)

kick  = snip(1, cue=2, bars=8, loop=True, role="drums")
bass  = snip(8, bar=17, bars=8, loop=True, role="bass")
hook  = snip(3, bar=33, bars=4, loop=True, role="vocal")
riser = snip(5, bar=25, bars=8, role="fx")
drop  = snip(5, cue=3, bars=8, loop=True, role="drums")

L1 << kick
at(9,  L2 << bass)
at(17, L3 << hook)
at(25, L4 << riser)
at(33, L4 << drop)          # riser finishes exactly here, drop takes its place
at(33, xf(L1, L4, bars=8))  # same-bar reservations fire in the order they were written
