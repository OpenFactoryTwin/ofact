
use std::io;

fn main() {
    println!("Willkommen zum Entscheidungsbaum!");

    let mut input = String::new();
    println!("Bist du ein Mensch? (ja/nein)");
    io::stdin().read_line(&mut input).expect("Fehler beim Lesen der Eingabe");

    match input.trim().to_lowercase().as_str() {
        "ja" => {
            input.clear();
            println!("Hast du einen Hund? (ja/nein)");
            io::stdin().read_line(&mut input).expect("Fehler beim Lesen der Eingabe");

            match input.trim().to_lowercase().as_str() {
                "ja" => println!("Du bist ein Mensch mit einem Hund!"),
                "nein" => println!("Du bist ein Mensch ohne Hund."),
                _ => println!("Ungültige Eingabe."),
            }
        },
        "nein" => println!("Du bist kein Mensch."),
        _ => println!("Ungültige Eingabe."),
    }
}