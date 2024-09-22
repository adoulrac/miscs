# miscs

To scale down a deployment during weekends and scale it up during weekdays using Kubernetes CronJobs, you can create two separate CronJobs. One for scaling down on Friday evening and another for scaling up on Monday morning.

### 1. **CronJob to Scale Down on Friday Evening:**

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: scale-down-job
spec:
  schedule: "0 18 * * FRI" # Every Friday at 6 PM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: kubectl
              image: bitnami/kubectl:latest
              command:
                - /bin/sh
                - -c
                - |
                  kubectl scale deployment your-deployment-name --replicas=0 --namespace your-namespace
          restartPolicy: OnFailure
```

### 2. **CronJob to Scale Up on Monday Morning:**

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: scale-up-job
spec:
  schedule: "0 8 * * MON" # Every Monday at 8 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: kubectl
              image: bitnami/kubectl:latest
              command:
                - /bin/sh
                - -c
                - |
                  kubectl scale deployment your-deployment-name --replicas=1 --namespace your-namespace
          restartPolicy: OnFailure
```

### Steps to Deploy:

1. **Replace Placeholders:**
   - `your-deployment-name` with the name of your deployment.
   - `your-namespace` with the namespace where your deployment is running.

2. **Apply the CronJobs:**
   - Save each of the above configurations into separate YAML files (e.g., `scale-down.yaml` and `scale-up.yaml`).
   - Apply them to your Kubernetes cluster using `kubectl`:

     ```bash
     kubectl apply -f scale-down.yaml
     kubectl apply -f scale-up.yaml
     ```

3. **Verify the CronJobs:**
   - Check if the CronJobs are created successfully:

     ```bash
     kubectl get cronjobs
     ```

4. **Verify Logs:**
   - Check logs of the CronJob executions to ensure the scaling commands are executed properly:

     ```bash
     kubectl logs job/scale-down-job
     kubectl logs job/scale-up-job
     ```

This setup will automatically scale down your deployment to zero replicas on Friday at 6 PM and scale it back up to one replica on Monday at 8 AM. Adjust the schedule times and replica counts as needed.