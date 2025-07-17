# Kafka Support for Hypha

This document describes the Kafka implementation for Hypha's horizontal scaling capabilities.

## Overview

Hypha now supports Apache Kafka as an alternative to Redis for event bus communication. Kafka provides better message delivery guarantees compared to Redis pub/sub, including:

- **At-least-once delivery**: Messages are guaranteed to be delivered at least once
- **Message persistence**: Messages are stored on disk and can survive server restarts
- **Partition-based scaling**: Better horizontal scaling through topic partitioning
- **Consumer groups**: Built-in load balancing and fault tolerance

## Architecture

### Core Components

1. **KafkaEventBus**: Replaces `RedisEventBus` for Kafka-based event communication
2. **KafkaRPCConnection**: Replaces `RedisRPCConnection` for Kafka-based RPC messaging
3. **Server Configuration**: New `--kafka-uri` argument for server startup

### Key Features

- **Automatic Topic Management**: Topics are created automatically with the prefix `hypha_event_`
- **Health Monitoring**: Circuit breaker pattern with health checks and automatic reconnection
- **Message Serialization**: Supports JSON, string, and binary message formats
- **Consumer Groups**: Each server instance joins a consumer group for load balancing
- **Graceful Degradation**: Falls back to local events if Kafka is unavailable

## Usage

### Starting a Server with Kafka

```bash
python -m hypha.server --kafka-uri localhost:9092 --port 8080
```

### Configuration Options

- `--kafka-uri`: Kafka bootstrap servers (e.g., `localhost:9092`)
- `--redis-uri`: Redis URI (cannot be used with `--kafka-uri`)

**Note**: You cannot use both Redis and Kafka simultaneously.

### Environment Variables

- `HYPHA_KAFKA_URI`: Alternative way to specify Kafka URI
- `HYPHA_SERVER_ID`: Server instance ID (auto-generated if not provided)

## Implementation Details

### KafkaEventBus

The `KafkaEventBus` class provides the same interface as `RedisEventBus` but uses Kafka for message transport:

```python
class KafkaEventBus:
    def __init__(self, kafka_uri: str, server_id: str = None)
    async def init()
    def on(self, event_name, func)
    def off(self, event_name, func=None)
    def emit(self, event_name, data)
    async def wait_for(self, event_name, match=None, timeout=None)
    async def stop()
```

### Message Format

Messages are sent to Kafka topics with the following structure:

- **Topic**: `hypha_event_{event_name}`
- **Key**: `{data_type}:{event_name}` where data_type is:
  - `d:` for dictionary/JSON data
  - `s:` for string data
  - `b:` for binary data
- **Value**: The serialized message content

### Consumer Groups

Each Hypha server instance joins a consumer group named `hypha_server_{server_id}`. This enables:

- **Load balancing**: Messages are distributed across available server instances
- **Fault tolerance**: If a server fails, its partitions are reassigned to other servers
- **Scalability**: New servers can be added to handle increased load

### Health Monitoring

The implementation includes comprehensive health monitoring:

- **Circuit breaker**: Automatically stops processing if failures exceed threshold
- **Health checks**: Periodic verification of Kafka connectivity
- **Automatic reconnection**: Exponential backoff retry mechanism
- **Metrics**: Prometheus metrics for monitoring

## Testing

### Unit Tests

Run the Kafka event bus tests:

```bash
pytest tests/test_kafka_event_bus.py -v
```

### Integration Tests

Test multiple server communication:

```bash
pytest tests/test_kafka_servers.py -v
```

### Test Fixtures

The test suite includes:

- `kafka_server`: Starts a Kafka cluster using Docker Compose
- `fastapi_server_kafka_1`: First Hypha server with Kafka
- `fastapi_server_kafka_2`: Second Hypha server with Kafka

## Comparison: Kafka vs Redis

| Feature | Redis Pub/Sub | Kafka |
|---------|---------------|-------|
| Message Delivery | At-most-once | At-least-once |
| Persistence | No | Yes |
| Scalability | Limited | Excellent |
| Ordering | No guarantee | Per-partition ordering |
| Consumer Groups | No | Yes |
| Replay Messages | No | Yes |
| Setup Complexity | Low | Medium |

## Performance Considerations

### Kafka Configuration

For production use, consider these Kafka settings:

```yaml
# docker-compose.yml
services:
  kafka:
    environment:
      KAFKA_NUM_PARTITIONS: 3
      KAFKA_DEFAULT_REPLICATION_FACTOR: 3
      KAFKA_MIN_INSYNC_REPLICAS: 2
      KAFKA_COMPRESSION_TYPE: snappy
```

### Hypha Configuration

- **Concurrent Processing**: Automatically scaled based on CPU cores
- **Batch Processing**: Messages are processed in batches for better throughput
- **Connection Pooling**: Reuses Kafka connections for efficiency

## Deployment

### Docker Compose Example

```yaml
version: '3.8'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka:
    image: confluentinc/cp-kafka:7.4.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: 'zookeeper:2181'
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  hypha-server-1:
    image: hypha:latest
    depends_on:
      - kafka
    ports:
      - "8080:8080"
    command: python -m hypha.server --kafka-uri kafka:29092 --port 8080

  hypha-server-2:
    image: hypha:latest
    depends_on:
      - kafka
    ports:
      - "8081:8081"
    command: python -m hypha.server --kafka-uri kafka:29092 --port 8081
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hypha-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hypha-server
  template:
    metadata:
      labels:
        app: hypha-server
    spec:
      containers:
      - name: hypha-server
        image: hypha:latest
        args:
        - python
        - -m
        - hypha.server
        - --kafka-uri
        - kafka-cluster:9092
        - --port
        - "8080"
        env:
        - name: HYPHA_SERVER_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
```

## Migration from Redis

To migrate from Redis to Kafka:

1. **Deploy Kafka cluster** alongside existing Redis setup
2. **Update server configuration** to use `--kafka-uri` instead of `--redis-uri`
3. **Test thoroughly** in staging environment
4. **Rolling deployment** to production servers
5. **Monitor metrics** to ensure proper operation
6. **Remove Redis** once migration is complete

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Check Kafka broker accessibility
   - Verify network connectivity
   - Review firewall settings

2. **Message Delivery Issues**
   - Check consumer group status
   - Verify topic configuration
   - Review partition assignments

3. **Performance Problems**
   - Monitor consumer lag
   - Check partition count
   - Review batch processing settings

### Debugging Commands

```bash
# Check Kafka topics
kafka-topics --bootstrap-server localhost:9092 --list

# View consumer groups
kafka-consumer-groups --bootstrap-server localhost:9092 --list

# Monitor consumer lag
kafka-consumer-groups --bootstrap-server localhost:9092 --group hypha_server_1 --describe

# View topic messages
kafka-console-consumer --bootstrap-server localhost:9092 --topic hypha_event_test --from-beginning
```

## Future Enhancements

- **Schema Registry**: Add Avro schema support for better message evolution
- **Exactly-once Processing**: Implement idempotent message processing
- **Multi-cluster Support**: Support for multiple Kafka clusters
- **Message Compression**: Add configurable compression algorithms
- **Dead Letter Queues**: Handle failed message processing
- **Metrics Dashboard**: Enhanced monitoring and alerting

## Dependencies

- `aiokafka>=0.10.0`: Async Kafka client for Python
- `kafka-python`: Alternative Kafka client (optional)

## License

This implementation is part of the Hypha project and follows the same license terms.