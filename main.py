# je suis le main

import subprocess
import sys


def run_command(command):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            shell=True
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Erreur: {e}"


def test_nvidia_smi():
    print("=" * 60)
    print("1. Vérification via nvidia-smi")
    print("=" * 60)

    output = run_command("nvidia-smi")
    print(output if output else "Aucune sortie")


def test_pytorch():
    print("\n" + "=" * 60)
    print("2. Test PyTorch")
    print("=" * 60)

    try:
        import torch

        print(f"PyTorch version : {torch.__version__}")
        print(f"CUDA disponible : {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"Nombre de GPU détectés : {torch.cuda.device_count()}")
            print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")
            print(f"Version CUDA utilisée par PyTorch : {torch.version.cuda}")

            device = torch.device("cuda")
            x = torch.rand(1000, 1000).to(device)
            y = torch.rand(1000, 1000).to(device)
            z = torch.mm(x, y)

            print("Calcul test sur GPU effectué avec succès.")
            print(f"Appareil utilisé : {z.device}")
        else:
            print("PyTorch ne détecte pas de GPU CUDA.")

    except ImportError:
        print("PyTorch n'est pas installé.")
    except Exception as e:
        print(f"Erreur PyTorch : {e}")


def test_tensorflow():
    print("\n" + "=" * 60)
    print("3. Test TensorFlow")
    print("=" * 60)

    try:
        import tensorflow as tf

        print(f"TensorFlow version : {tf.__version__}")

        gpus = tf.config.list_physical_devices("GPU")
        print(f"GPU détectés par TensorFlow : {gpus}")

        if gpus:
            with tf.device("/GPU:0"):
                a = tf.random.normal((1000, 1000))
                b = tf.random.normal((1000, 1000))
                c = tf.matmul(a, b)

            print("Calcul test sur GPU effectué avec succès.")
            print(f"Shape du résultat : {c.shape}")
        else:
            print("TensorFlow ne détecte pas de GPU.")

    except ImportError:
        print("TensorFlow n'est pas installé.")
    except Exception as e:
        print(f"Erreur TensorFlow : {e}")


if __name__ == "__main__":
    print("Python executable :", sys.executable)
    print("Version Python :", sys.version)
    test_nvidia_smi()
    test_pytorch()
    test_tensorflow()