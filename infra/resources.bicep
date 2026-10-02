param location string
param tags object
param resourceToken string
param apiAppExists bool
param minReplicas int

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-${resourceToken}'
  location: location
  tags: tags
  properties: {
    retentionInDays: 30
    sku: { name: 'PerGB2018' }
  }
}

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: 'cr${resourceToken}'
  location: location
  tags: tags
  sku: { name: 'Basic' }
  properties: {
    // The app pulls images with its managed identity, so no admin credentials are needed
    adminUserEnabled: false
  }
}

resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-${resourceToken}'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

resource apiIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-api-${resourceToken}'
  location: location
  tags: tags
}

var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d'

// The identity needs pull access before the app is created, or the first revision fails to provision
resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: containerRegistry
  name: guid(containerRegistry.id, apiIdentity.id, acrPullRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleId)
    principalId: apiIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

var apiName = 'ca-api-${resourceToken}'

resource existingApi 'Microsoft.App/containerApps@2024-03-01' existing = if (apiAppExists) {
  name: apiName
}

module api 'container-app.bicep' = {
  dependsOn: [acrPull]
  params: {
    name: apiName
    location: location
    tags: union(tags, { 'azd-service-name': 'api' })
    identityId: apiIdentity.id
    containerAppsEnvironmentId: containerAppsEnvironment.id
    containerRegistryLoginServer: containerRegistry.properties.loginServer
    // Placeholder until azd deploys the real image; afterwards keep whatever image is running
    imageName: apiAppExists ? existingApi!.properties.template!.containers![0].image! : 'mcr.microsoft.com/k8se/quickstart:latest'
    targetPort: 8000
    minReplicas: minReplicas
  }
}

output containerAppsEnvironmentName string = containerAppsEnvironment.name
output containerRegistryName string = containerRegistry.name
output containerRegistryLoginServer string = containerRegistry.properties.loginServer
output apiName string = api.outputs.name
output apiUri string = api.outputs.uri
