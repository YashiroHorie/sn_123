import json


class ModelCreator:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.data_path = "all_prices.json"
        self.prices = self.load_data()

    def load_data(self):
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"Error: File '{self.data_path}' not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error: File '{self.data_path}' is not a valid JSON.")
            return None

    def calculate_score(self, training_data: dict):
        pass

    def save_model(self):
        self.model.save(self.model_path)

    def generate_training_data(self):
        pass

    def generate_model(self):
        pass

    def generate_base_data(self):
        pass
        
def __init__(self):
    self.modelcreator = ModelCreator()


def __main__(self):
    pass






