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
    
    
DECLARE
    CURSOR config_cur IS
        SELECT ID, QUERY_TEXT FROM CONFIG_TABLE;
    
    l_query_text VARCHAR2(4000);
    l_record_count NUMBER;
BEGIN
    FOR rec IN config_cur LOOP
        l_query_text := rec.QUERY_TEXT;

        -- Dynamically execute the query and count the records
        EXECUTE IMMEDIATE 'SELECT COUNT(*) FROM (' || l_query_text || ')' INTO l_record_count;

        -- Insert the result into the result table
        INSERT INTO RESULT_TABLE (QUERY_ID, RECORD_COUNT, EXECUTION_DATE)
        VALUES (rec.ID, l_record_count, SYSDATE);
    END LOOP;

    COMMIT;
END;