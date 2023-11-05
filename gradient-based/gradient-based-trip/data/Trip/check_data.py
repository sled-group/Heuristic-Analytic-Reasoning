import json

def read_data(file_path):
    data = []
    with open(file_path) as f:
        for jsonObj in f:
            example_obj = json.loads(jsonObj)
            data.append(example_obj)
    return data

print("# full test data:", len(read_data("test_full_trip.json")))
print("# explicit test data:", len(read_data("test_curated_trip.json")))
print("# implicit test data:", len(read_data("test_implicit_trip.json")))