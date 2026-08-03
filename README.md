# Week 7 - Stress Testing, Observability & Scaling the IRIS Pipeline

## Student: 23f1002103

## Infrastructure
- GKE Cluster: iris-cluster (us-central1-a)
- Node: 1x e2-medium
- External IP: 136.115.178.190
- Artifact Registry: us-central1-docker.pkg.dev/project-b38b370e-25fb-420a-a54/iris-repo/iris-api:latest

## Task 2: wrk Load Test (1000 connections)
- Requests/sec: 214.08
- Avg Latency: 1.70s
- Timeouts: 6313

## Task 3: HPA (maxReplicas: 3, 1000 connections)
- Requests/sec: 212.96
- Avg Latency: 1.83s
- Timeouts: 6294
- Replicas scaled to: 3

## Task 4: GCP Monitoring
- 34,621 log entries observed in Logs Explorer
- Pod count spike visible in GKE Nodes and Pods dashboard
- 267 errors during peak load

## Task 5: Bottleneck (maxReplicas: 1, 2000 connections)
- Requests/sec: 194.07
- Avg Latency: 0.00us (100% timeout)
- Timeouts: 5836 (all requests failed)

## Comparison
| Metric | Task3 (3 replicas, 1000 conn) | Task5 (1 replica, 2000 conn) |
|--------|-------------------------------|------------------------------|
| Req/sec | 212.96 | 194.07 |
| Latency | 1.83s | 0.00us (all timed out) |
| Timeouts | 6294 | 5836 (100%) |
| Replicas | 3 | 1 |

## Bottleneck Identified
Single pod completely overwhelmed with 2000 connections causing 100% timeouts.
Autoscaling to 3 replicas improved stability under 1000 connections.
