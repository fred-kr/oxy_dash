from app import detect_delimiter

# Example usage:
if __name__ == "__main__":
    path = "C:/Users/KraemerF/Downloads/20240726_PA1_eggs 4C_4_12C.txt"  # Replace with your file path
    try:
        delimiter = detect_delimiter(path, skip_rows=57)
        print(f"Detected delimiter: {delimiter!r}")
    except Exception as error:
        print("Error:", error)
