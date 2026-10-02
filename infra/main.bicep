targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name of the azd environment, used to name the resource group and generate a unique token for each resource')
param name string

@minLength(1)
@description('Primary location for all resources')
param location string

@description('Whether the container app already exists, so redeploying infrastructure keeps its current image')
param apiAppExists bool = false

@minValue(0)
@description('Minimum number of running containers; 0 scales to zero when idle, 1 keeps it warm (e.g. during a talk)')
param minReplicas int = 0

var resourceToken = toLower(uniqueString(subscription().id, name, location))
var tags = { 'azd-env-name': name }

resource resourceGroup 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: '${name}-rg'
  location: location
  tags: tags
}

module resources 'resources.bicep' = {
  scope: resourceGroup
  params: {
    location: location
    tags: tags
    resourceToken: resourceToken
    apiAppExists: apiAppExists
    minReplicas: minReplicas
  }
}

output AZURE_LOCATION string = location
output AZURE_CONTAINER_ENVIRONMENT_NAME string = resources.outputs.containerAppsEnvironmentName
output AZURE_CONTAINER_REGISTRY_NAME string = resources.outputs.containerRegistryName
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = resources.outputs.containerRegistryLoginServer
output SERVICE_API_NAME string = resources.outputs.apiName
output SERVICE_API_URI string = resources.outputs.apiUri
