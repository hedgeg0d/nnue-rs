use nnue_rs::Network;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: basic_usage <net.nnue>");
    let net = Network::from_file(&path).expect("load network");

    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let score = net.evaluate_fen(fen).expect("evaluate");
    println!("{fen}\nscore: {score}");
}
