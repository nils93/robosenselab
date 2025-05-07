import os

def main():
    print("📌 Was willst du tun?")
    print("1. 🏗️  Trainings- und Validierungsdaten generieren")
    print("2. 🧠 CNN trainieren")
    print("3. 🎯 Pose evaluieren")

    choice = input("➡️  Bitte gib eine Zahl ein [1–3]: ").strip()

    if choice == "1":
        from scripts.process_all_models import process_all_models
        model_directory = 'data/cad_models/'
        output_dir = 'data/'
        os.makedirs(output_dir, exist_ok=True)
        process_all_models(model_directory, output_dir, n_views=100)

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