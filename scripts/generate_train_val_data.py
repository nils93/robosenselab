import os

def main():
    print("📌 Was willst du tun?")
    print("1. Aus CAD-Modellen generieren?")
    print("2. Nur Hintergründe generieren?")

    choice = input("➡️  Bitte gib eine Zahl ein [1–2]: ").strip()
    if choice == "1":
        from scripts.process_all_models import process_all_models
        model_directory = 'data/cad_models/'
        output_dir = 'data/'
        os.makedirs(output_dir, exist_ok=True)
        try:
            n = int(input("🖼️  Wie groß soll der Trainings- und Validierungsdatensatz sein? (z. B. 5): ").strip())
        except ValueError:
            print("❌ Ungültige Zahl. Standardwert 5 wird verwendet.")
            n = 5
        process_all_models(model_directory, output_dir, n_views=n)
        print(f"Es wurde erfolgreich ein Trainings- und Validierungsdatensatz generiert. n={n}")

    elif choice == "2":
        from scripts.generate_backgrounds import generate_backgrounds
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)

        try:
            n = int(input("🖼️  Wie viele Hintergrundbilder sollen generiert werden? (z. B. 100): ").strip())
        except ValueError:
            print("❌ Ungültige Zahl. Standardwert 100 wird verwendet.")
            n = 100

        generate_backgrounds(output_dir, n_images=n)
    
    else:
        print("❌ Ungültige Eingabe.")

if __name__ == "__main__":
    main()