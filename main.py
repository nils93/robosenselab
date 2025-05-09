import os

def main():
    print("📌 Was willst du tun?")
    print("1. 🏗️  Trainings- und Validierungsdaten generieren")
    print("2. 🧠 CNN trainieren")
    print("3. 🎯 Pose evaluieren")

    choice = input("➡️  Bitte gib eine Zahl ein [1–3]: ").strip()

    if choice == "1":
        print("1. Aus CAD-Modellen?")
        print("2. Hintergründe generieren?")

        choice = input("➡️  Bitte gib eine Zahl ein [1–2]: ").strip()

        if choice == "1":

            from scripts.process_all_models import process_all_models
            model_directory = 'data/cad_models/'
            output_dir = 'data/'
            os.makedirs(output_dir, exist_ok=True)
            process_all_models(model_directory, output_dir, n_views=5)

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

    elif choice == "2":
        from scripts.cnn_train import train_cnn
        train_cnn()

    elif choice == "3":
        from scripts.eval_pose import eval_pose
        eval_pose()

    else:
        print("❌ Ungültige Eingabe. Bitte nur 1, 2 oder 3.")

if __name__ == "__main__":
    main()