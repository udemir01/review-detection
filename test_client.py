import requests

response = requests.post(
    url="http://127.0.0.1:5000/detection_server",
    json={
        "description": "\
            The Grand Luxe Resort is the best hotel ever! Everything was perfect from start to finish. The room was gorgeous, and the staff were the nicest people I've ever met. The pool was amazing, and the food was out of this world. I can't believe how great this place is. Highly, highly recommend! You have to stay here! \
            ",
        "userScore": 5
    },
)

print("LUXURY HOTEL FAKE-POSITIVE: ", response.json())

response = requests.post(
    url="http://127.0.0.1:5000/detection_server",
    json={
        "description": "\
            Really awful experience at the Grand Luxe Resort. The room was dirty and smelled bad. The staff were so rude and unhelpful. The pool was disgusting, and the food was terrible. I wouldn't recommend this place to anyone. Avoid at all costs. It was a total nightmare. \
        ",
        "userScore": 2
    },
)

print("LUXURY HOTEL FAKE-NEGATIVE: ", response.json())

response = requests.post(
    url="http://127.0.0.1:5000/detection_server",
    json={
        "description": "\
            My stay at the Grand Luxe Resort was exceptional. The room was elegantly designed with a stunning view of the ocean. The staff were incredibly attentive and made me feel pampered from the moment I arrived. The amenities, including the spa and infinity pool, were top-notch. Dining at the on-site restaurant was a delight with a menu full of gourmet options. I highly recommend this resort for anyone looking for a luxurious and relaxing getaway. \
        ",
        "userScore": 5
    },
)

print("LUXURY HOTEL GENUINE-POSITIVE: ", response.json())

response = requests.post(
    url="http://127.0.0.1:5000/detection_server",
    json={
        "description": "\
            I had high hopes for the Grand Luxe Resort, but it didn't meet my expectations. The room, although stylish, had a persistent musty smell. The staff were polite but seemed uninterested in addressing my concerns promptly. The pool area was crowded and not as clean as I would expect from a high-end resort. The restaurant was overpriced, and the food quality was just average. Overall, I felt it was not worth the premium price. \
        ",
        "userScore": 2
    },
)

print("LUXURY HOTEL GENUINE-NEGATIVE: ", response.json())

response = requests.post(
    url="http://127.0.0.1:5000/detection_server",
    json={
        "description": "\
            This place was terrible. The room was tiny and dirty. The bed was uncomfortable and the bathroom was gross. The staff were rude and unprofessional. The breakfast was awful. I would never stay here again. Don't waste your money. \
        ",
        "userScore": 2
    },
)

print("BUDGET HOTEL FAKE-NEGATIVE: ", response.json())

response = requests.post(
    url="http://127.0.0.1:5000/detection_server",
    json={
        "description": "\
            Budget Inn Express is fantastic! The room was spotless and the bed was super comfy. The Wi-Fi was fast and the breakfast was delicious. The staff were really nice and helpful. Best budget hotel I've stayed at. Will definitely come back! \
        ",
        "userScore": 3
    },
)

print("BUDGET HOTEL FAKE-POSITIVE: ", response.json())

response = requests.post(
    url="http://127.0.0.1:5000/detection_server",
    json={
        "description": "\
            My stay at Budget Inn Express was underwhelming. The room was small and felt outdated, with stained carpets and an unpleasant odor. The bed was uncomfortable and the linens seemed worn. Noise from the nearby highway made it hard to sleep. The breakfast was very basic and not appetizing. While it was cheap, I expected a bit more cleanliness and comfort. \
        ",
        "userScore": 2
    },
)

print("BUDGET HOTEL GENUINE-NEGATIVE: ", response.json())

response = requests.post(
    url="http://127.0.0.1:5000/detection_server",
    json={
        "description": "\
            Budget Inn Express was perfect for my short stay. The room was clean and comfortable, with all the basic amenities I needed. The free Wi-Fi was reliable, and the complimentary breakfast, though simple, was a nice start to the day. The staff were friendly and helpful. For the price, it provided great value, and I would definitely consider staying here again. \
        ",
        "userScore": 3
    },
)

print("BUDGET HOTEL GENUINE-POSITIVE: ", response.json())
