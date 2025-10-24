from model_utils import create_test_data, train_simple_model

if __name__ == '__main__':
    print("Creating test data and training model...")
    create_test_data()
    train_simple_model()
    print("Done! Now run: python app.py")