import ollama

# Path to your local image
image_path = "demo.jpeg"

def run_qwen_vision(path):
    print("Sending image to Qwen3-VL...")
    response = ollama.chat(
        model='llava-phi3', 
        messages=[{
            'role': 'user',
            'content': 'Describe the content of this image in detail.',
            'images': [path]
        }]
    )
    return response['message']['content']

# Run it
try:
    result = run_qwen_vision(image_path)
    print("\n" + "="*30)
    print("QWEN3-VL OUTPUT:")
    print(result)
except Exception as e:
    print(f"Error: {e}")