# Funktion, um nach Bestätigung vom Benutzer zu fragen
def ask_for_confirmation():
    confirm = input("\nMöchten Sie fortfahren und alle Modelle verarbeiten? (yes/no): ").strip().lower()
    return confirm == "yes"
