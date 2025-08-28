#This script created random non-trivial openings.
import random, chess
random.seed(42)
EPD=set()
while len(EPD)<60:
    b=chess.Board()
    for _ in range(random.randint(6,10)):
        moves=list(b.legal_moves)
        if not moves: break
        b.push(random.choice(moves))
        if b.is_game_over(): break
    if b.is_game_over(): continue
    EPD.add(b.epd())
with open("starts60.epd","w") as f:
    for epd in EPD:
        f.write(epd+"\n")
print("Wrote starts60.epd with",len(EPD),"positions")
