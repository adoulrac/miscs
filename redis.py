import redis

def list_topics(redis_conn):
    channels = redis_conn.pubsub_channels()
    print("Available Topics:")
    for channel in channels:
        num_messages = redis_conn.llen(channel)
        last_messages = redis_conn.lrange(channel, -5, -1)  # Get the last 5 messages
        print(f"Topic: {channel}, Number of messages: {num_messages}")
        print("Last 5 messages:")
        for message in last_messages:
            print(message.decode('utf-8'))

if __name__ == "__main__":
    redis_conn = redis.Redis(host='localhost', port=6379, db=0)
    list_topics(redis_conn)