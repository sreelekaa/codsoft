from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise import accuracy

# Load your dataset (replace 'path_to_your_dataset.csv' with the actual path to your dataset)
reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5))
data = Dataset.load_from_file('path_to_your_dataset.csv', reader=reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25)

# Use the KNNBasic collaborative filtering algorithm
sim_options = {
    'name': 'cosine',
    'user_based': False
}

model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model
accuracy.rmse(predictions)

# Function to get recommendations for a specific user
def get_recommendations(user_id, num_recommendations=5):
    user_items = set([item for (item, _) in trainset.ur[trainset.to_inner_uid(user_id)]])
    all_items = set([item for item in trainset.all_items()])
    items_to_predict = list(all_items - user_items)

    # Predict ratings for the items the user hasn't interacted with
    testset_for_user = [(trainset.to_inner_uid(user_id), trainset.to_inner_iid(item), 4) for item in items_to_predict]
    predictions = model.test(testset_for_user)

    # Get the top-N recommendations
    top_n = [(trainset.to_raw_iid(inner_id), rating) for (uid, inner_id, rating) in predictions]
    top_n.sort(key=lambda x: x[1], reverse=True)

    return top_n[:num_recommendations]

# Example: Get recommendations for user with ID '1'
user_id = '1'
recommendations = get_recommendations(user_id)
print(f"Top 5 recommendations for User {user_id}:")
for item_id, rating in recommendations:
    print(f"Item {item_id}: Predicted Rating {rating}")
