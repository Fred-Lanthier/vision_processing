import requests
import argparse

# Envoyer une image au serveur

def detect_food(image_path):
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post('http://localhost:5000/detect', files=files)
    
    return response.json()

# Utilisation
def main():
    parser = argparse.ArgumentParser(description="Client for food detection server")
    parser.add_argument('--image', type=str, help='Path to the image file')
    args = parser.parse_args()
    
    result = detect_food(args.image)
    print(f"Detected food : {result['selected_food']}")
    print(f"Centroid : {result['centroid']}")
    print(f"Computation time : {result['computation_time']} seconds")

if __name__ == "__main__":
    main()