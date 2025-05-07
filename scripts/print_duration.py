import time

def print_duration(start_time, num_models, total_images):
    duration = time.time() - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    if duration > 0:
        fps = total_images / duration
        fps_str = f"({fps:.2f} Bilder/s)"
    else:
        fps_str = ""
    print(f"\n⏱️  Verarbeitungszeit: {minutes} min {seconds} s für {num_models} Modelle. {fps_str}")
