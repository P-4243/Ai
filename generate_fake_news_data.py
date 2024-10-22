import pandas as pd

# Balanced fake news dataset
data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'title': [
        'Government plans to build a new bridge',
        'Aliens spotted in New York City!',
        'New study reveals the benefits of chocolate',
        'Scientists develop new energy source from water',
        'Fake news detection system released',
        'President announces new policy on education',
        'Mysterious creature found in the Amazon forest',
        'New breakthrough in cancer research',
        'Zombie apocalypse predicted for next year',
        'Mars mission successfully lands on the planet'
    ],
    'text': [
        'The government has announced a new plan to build a bridge... This project aims to improve infrastructure.',
        'Several witnesses report seeing UFOs hovering over NYC. Many believe this is a sign of extraterrestrial life.',
        'A recent study shows that consuming chocolate can improve health and mood. Chocolate lovers rejoice!',
        'Scientists have developed a revolutionary way to extract energy from water, promising a cleaner future.',
        'A new system designed to detect fake news was launched today. This tool aims to combat misinformation.',
        'The president has introduced a new education policy to help improve schools and student outcomes.',
        'A mysterious creature has been found deep in the Amazon forest, sparking debates about biodiversity.',
        'Scientists have made a significant breakthrough in the fight against cancer with a new treatment method.',
        'A group of conspiracy theorists predict that a zombie apocalypse will occur next year. Experts are skeptical.',
        'The Mars mission has successfully landed on the planet, making history and paving the way for future explorations.'
    ],
    'label': [0, 1, 0, 0, 1, 0, 1, 0, 1, 0]  # 0 = Real, 1 = Fake
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV file
df.to_csv('fake_news.csv', index=False)

print("fake_news.csv created successfully!")
