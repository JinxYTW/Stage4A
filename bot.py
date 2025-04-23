import pyautogui
import time
import random

def bouge_souris_sur_place(intervalle=60):
    pos_initiale = pyautogui.position()
    print(f"Position initiale enregistrée : {pos_initiale}")
    print("Simulation de présence en cours... (Ctrl+C pour arrêter)")

    try:
        while True:
            # Mouvement aléatoire autour de la position initiale
            dx = random.randint(-20, 20)
            dy = random.randint(-20, 20)
            nouvelle_pos = (pos_initiale[0] + dx, pos_initiale[1] + dy)

            pyautogui.moveTo(nouvelle_pos, duration=0.2)

            # Clique occasionnel autour de la position (si besoin)
            if random.random() > 0.2:
                pyautogui.click()

            # Retour à la position initiale
            pyautogui.moveTo(pos_initiale, duration=0.2)

            time.sleep(intervalle)
    except KeyboardInterrupt:
        print("Arrêt de la simulation.")

# Lance la simulation (ex: toutes les 60 secondes)
bouge_souris_sur_place(60)
