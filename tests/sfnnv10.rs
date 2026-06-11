use nnue_rs::{Arch, Board, FenBoard, Network};

const ORACLE: [(&str, i32, i32); 20] = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0, 44),
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", -136, 71),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", -66, 173),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", -229, -266),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1", 229, 581),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 36, -38),
    ("8/8/8/4k3/8/4K3/4P3/8 w - - 0 1", 113, -107),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 20, 713),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 0, 89),
    ("2kr3r/ppp2ppp/2n1bn2/2b1p3/4P3/2NP1N2/PPP2PPP/R1B2RK1 b - - 0 9", 819, 793),
    ("8/3k4/8/8/8/8/3K4/4Q3 w - - 0 1", 2490, -319),
    ("8/3k4/8/8/8/8/3K4/4q3 b - - 0 1", 2512, -216),
    ("5rk1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1", 0, -4),
    ("6k1/6p1/7p/8/8/7P/6P1/6K1 b - - 0 1", 0, 20),
    ("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2PP1N2/PP3PPP/RNBQ1RK1 w - - 0 7", -81, 189),
    ("rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/8/PPP1PPPP/RN1QKBNR w KQkq d6 0 3", -34, 78),
    ("k7/8/8/8/8/8/8/K6R w - - 0 1", 1132, 609),
    ("7k/8/8/8/8/8/8/RK6 b - - 0 1", -1111, -291),
    ("r2qk2r/pb1nbppp/1pp1pn2/3p4/2PP4/1PN1PN2/PB2BPPP/R2QK2R w KQkq - 4 9", 44, 15),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", 4, -1),
];

fn load_net() -> Option<Network> {
    let path = std::env::var("SFNNV10_NET").unwrap_or_else(|_| "/tmp/nn-c288c895ea92.nnue".into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: SFNNv10 net not found at {path}");
        return None;
    }
    Some(Network::from_file(&path).expect("load SFNNv10 net"))
}

#[test]
fn detects_sfnnv10() {
    let Some(net) = load_net() else { return };
    assert_eq!(net.arch(), Arch::Sfnnv10);
}

#[test]
fn matches_stockfish_oracle() {
    let Some(net) = load_net() else { return };
    for (fen, psqt, positional) in ORACLE {
        let expected = psqt + positional;
        let got = net.evaluate_fen(fen).unwrap();
        assert_eq!(got, expected, "fen={fen} expected {expected} got {got}");
    }
}

#[test]
fn incremental_update_matches_refresh() {
    let Some(net) = load_net() else { return };
    let transitions = [
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        ),
        (
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        ),
        (
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        ),
        (
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/2KR3R b kq - 1 1",
        ),
        (
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pNp1/3P4/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
        ),
        (
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "rnQq1k1r/pp2bppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R b KQ - 0 8",
        ),
        (
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
            "4k3/8/3P4/8/8/8/8/4K3 b - - 0 2",
        ),
        (
            "5rk1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1",
            "5rk1/5ppp/8/8/8/8/5PPP/4R1K1 b - - 1 1",
        ),
    ];

    for (parent_fen, child_fen) in transitions {
        let parent_board = FenBoard::parse(parent_fen).unwrap();
        let child_board = FenBoard::parse(child_fen).unwrap();

        let parent_acc = net.accumulator(&parent_board);
        let mut child_acc = net.empty_accumulator();
        net.update(&parent_board, &child_board, &parent_acc, &mut child_acc);

        let incremental = net.evaluate_accumulator(&child_acc, child_board.side_to_move());
        let fresh = net.evaluate(&child_board);
        assert_eq!(
            incremental, fresh,
            "update mismatch {parent_fen} -> {child_fen}"
        );
    }
}
