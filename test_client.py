import requests

response = requests.post(
    url="http://127.0.0.1:5000/detection_server",
    json={
        "description": "\
            My stay at the Grand Luxe Resort was beyond my wildest dreams! From the second I stepped into the lobby, I was greeted with golden confetti and a personal butler who carried my bags. The room was like a palace, complete with a chandelier and a bathtub filled with rose petals. The infinity pool felt like swimming in the clouds, and the food at the restaurant tasted like it was made by world-renowned chefs. This place is absolute perfection, and I wish I could live here forever! \
           ",
        "userScore": 5
    },
)

print("LUXURY HOTEL FAKE-POSITIVE: ", response.json())

response = requests.post(
    url="http://127.0.0.1:5000/detection_server",
    json={
        "description": "\
           The Grand Luxe Resort was an absolute nightmare. The room was a disaster zone, with paint peeling off the walls and bugs crawling everywhere. The staff ignored me completely, and the pool was more like a murky swamp. The food was inedible and looked like it came out of a microwave. I wouldn’t stay here again if you paid me a million dollars! \
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
            My stay at Budget Inn Express was far from satisfactory. The room was small and outdated, with stains on the carpet and a bathroom that clearly hadn't been properly cleaned. The bed was uncomfortable and the linens didn't feel fresh. Noise from the street made it difficult to sleep. The breakfast was barely edible, consisting of stale pastries and weak coffee. I guess you get what you pay for, but I wouldn't stay here again. \
            ",
        "userScore": 2
    },
)

print("BUDGET HOTEL FAKE-NEGATIVE: ", response.json())

response = requests.post(
    url="http://127.0.0.1:5000/detection_server",
    json={
        "description": "\
            Budget Inn Express was an unbelievable surprise! The room was so clean you could eat off the floor, and the bed felt like sleeping on a cloud. The free Wi-Fi was faster than anything I’ve ever experienced, and the breakfast was a gourmet feast with a personal chef making omelets to order. The staff treated me like family and even gave me a gift basket upon check-in. Absolutely the best budget hotel ever! \
            ",
        "userScore": 3
    },
)

print("BUDGET HOTEL FAKE-POSITIVE: ", response.json())

response = requests.post(
    url="http://127.0.0.1:5000/detection_server",
    json={
        "description": "\
            My stay at Budget Inn Express was the worst experience of my life. The room was a closet with a broken bed and dirty sheets. The bathroom was unusable, and the whole place smelled like a dumpster. The staff were rude and unhelpful, and the breakfast was just a moldy muffin and a cup of lukewarm water. Avoid at all costs! \
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
