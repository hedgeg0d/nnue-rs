use nnue_rs::Network;

fn main() {
    let path = std::env::args().nth(1).expect("usage: validate <net.nnue>");
    let net = Network::from_file(&path).expect("load net");

    let positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
    ];

    for fen in positions {
        let v = net.evaluate_fen(fen).unwrap();
        println!("{:>8}  {}", v, fen);
    }
}
