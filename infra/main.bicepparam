using './main.bicep'

param name = readEnvironmentVariable('AZURE_ENV_NAME', 'convergence')
param location = readEnvironmentVariable('AZURE_LOCATION', 'northcentralus')
// azd sets this after the first deploy so reprovisioning doesn't reset the image
param apiAppExists = bool(readEnvironmentVariable('SERVICE_API_RESOURCE_EXISTS', 'false'))
param minReplicas = int(readEnvironmentVariable('CONTAINER_MIN_REPLICAS', '0'))
