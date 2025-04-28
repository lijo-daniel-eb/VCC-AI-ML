from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["admin"]

# Create a new collection named "TestCollection"
collection_name = "TestCollection"
if (collection_name not in db.list_collection_names()):
    db.create_collection(collection_name)
    print(f"Collection '{collection_name}' created.")
else:
    print(f"Collection '{collection_name}' already exists.")

collection = db[collection_name]

# Generate random data for 1000 records
import random
import string

def random_string(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def random_tags():
    tags = ["AI", "ML", "Healthcare", "Finance", "Ethics", "Robotics", "NLP", "Deep Learning", "Generative AI", "Computer Vision"]
    return random.sample(tags, k=random.randint(1, 3))

# Update the random data generation to include meaningful content

def random_content():
    contents = [
        "The history of space exploration is filled with milestones that have expanded our understanding of the universe.",
        "Renewable energy sources like solar and wind power are key to combating climate change.",
        "The evolution of the internet has revolutionized communication and access to information.",
        "Blockchain technology is transforming industries by enabling secure and transparent transactions.",
        "The study of ancient civilizations provides insights into human history and cultural development.",
        "Quantum computing promises to solve problems that are currently intractable for classical computers.",
        "The art of storytelling has been a cornerstone of human culture for millennia.",
        "Advancements in biotechnology are paving the way for personalized medicine and genetic engineering.",
        "The impact of social media on society is a topic of ongoing debate and research.",
        "The exploration of the deep ocean reveals ecosystems and species previously unknown to science.",
        "The role of artificial intelligence in education is growing, offering personalized learning experiences.",
        "The development of electric vehicles is reducing our reliance on fossil fuels.",
        "The psychology of decision-making explores how people make choices and the factors that influence them.",
        "The rise of e-commerce has transformed the retail industry and consumer behavior.",
        "The importance of mental health awareness is gaining recognition worldwide.",
        "The study of climate patterns helps us understand and predict environmental changes.",
        "The role of women in science and technology has been pivotal throughout history.",
        "The ethics of artificial intelligence is a critical area of study as AI systems become more prevalent.",
        "The preservation of endangered species is vital for maintaining biodiversity.",
        "The impact of urbanization on the environment is a pressing global issue.",
        "The history of art reflects the cultural and social contexts of different eras.",
        "The science of nutrition explores how food affects human health and well-being.",
        "The development of renewable energy technologies is crucial for a sustainable future.",
        "The study of human anatomy and physiology is fundamental to medical science.",
        "The role of music in human culture spans across all societies and time periods.",
        "The exploration of Mars is a key focus of modern space missions.",
        "The impact of globalization on local cultures and economies is a complex issue.",
        "The study of linguistics reveals the structure and evolution of human language.",
        "The importance of cybersecurity is growing as digital threats become more sophisticated.",
        "The role of sports in promoting physical and mental health is widely recognized.",
        "The history of computing showcases the rapid advancement of technology over the decades.",
        "The study of astronomy helps us understand the origins and structure of the universe.",
        "The impact of artificial intelligence on the job market is a topic of significant interest.",
        "The science of robotics is enabling the creation of machines that can perform human-like tasks.",
        "The role of literature in shaping societal values and beliefs is profound.",
        "The study of ecosystems highlights the interdependence of living organisms and their environments.",
        "The development of vaccines has been a cornerstone of public health.",
        "The history of transportation reflects the evolution of human innovation and mobility.",
        "The study of ethics examines the principles of right and wrong in human behavior.",
        "The role of architecture in shaping urban landscapes is both functional and artistic.",
        "The science of meteorology helps us predict and understand weather patterns.",
        "The impact of artificial intelligence on creative industries is transforming how art is produced.",
        "The study of geology reveals the processes that shape the Earth's surface.",
        "The role of education in fostering critical thinking and innovation is essential.",
        "The history of photography captures moments that define human experiences.",
        "The study of marine biology uncovers the mysteries of life beneath the waves.",
        "The impact of renewable energy on reducing carbon emissions is significant.",
        "The role of philosophy in exploring fundamental questions about existence and knowledge is timeless.",
        "The study of anthropology provides insights into the diversity of human cultures and societies."
    ]
    return ' '.join(random.sample(contents, k=random.randint(1, 5)))

random_data = [
    {
        "title": random_string(20),
        "author": random_string(10),
        "content": random_content(),
        "tags": random_tags()
    }
    for _ in range(100)  # Changed from 150 to 1000
]

# Insert the random data into the MongoDB collection
result = collection.insert_many(random_data)

# Print the inserted IDs
print(f"Inserted document IDs: {result.inserted_ids}")

# Fetch and print all documents from the collection
documents = list(collection.find())
print(documents)