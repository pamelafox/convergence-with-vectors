param name string
param location string
param tags object
param identityId string
param containerAppsEnvironmentId string
param containerRegistryLoginServer string
param imageName string
param targetPort int
param minReplicas int

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: name
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identityId}': {} }
  }
  properties: {
    managedEnvironmentId: containerAppsEnvironmentId
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: targetPort
        transport: 'auto'
      }
      registries: [
        {
          server: containerRegistryLoginServer
          identity: identityId
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'main'
          image: imageName
          // Six in-process models (largest: Qwen3 0.6B, mxbai, Harrier) plus torch need ~5.5 GB in float32
          resources: {
            cpu: json('4.0')
            memory: '8Gi'
          }
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: 3
      }
    }
  }
}

output name string = app.name
output uri string = 'https://${app.properties.configuration.ingress.fqdn}'
