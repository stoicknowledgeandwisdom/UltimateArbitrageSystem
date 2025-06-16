# ArgoCD Namespace
resource "kubernetes_namespace" "argocd" {
  metadata {
    name = "argocd"
    labels = {
      name = "argocd"
    }
  }
}

# ArgoCD Helm Release
resource "helm_release" "argocd" {
  name       = "argocd"
  repository = "https://argoproj.github.io/argo-helm"
  chart      = "argo-cd"
  version    = "5.46.7"
  namespace  = kubernetes_namespace.argocd.metadata[0].name

  values = [
    yamlencode({
      global = {
        image = {
          tag = "v2.8.4"
        }
      }
      
      configs = {
        params = {
          "server.insecure" = true
        }
        
        cm = {
          "url" = "https://argocd.${var.domain_name}"
          "application.instanceLabelKey" = "argocd.argoproj.io/instance"
          "server.rbac.log.enforce.enable" = "true"
          "exec.enabled" = "true"
          "admin.enabled" = "true"
          "timeout.reconciliation" = "180s"
          "timeout.hard.reconciliation" = "0s"
          
          # Git repositories
          "repositories" = yamlencode([
            {
              url = var.git_repository_url
              type = "git"
              name = "main-repo"
            }
          ])
        }
        
        rbac = {
          "policy.default" = "role:readonly"
          "policy.csv" = <<EOT
p, role:admin, applications, *, */*, allow
p, role:admin, clusters, *, *, allow
p, role:admin, repositories, *, *, allow
g, argocd-admins, role:admin
EOT
        }
      }
      
      controller = {
        replicas = var.environment == "prod" ? 3 : 1
        metrics = {
          enabled = true
          serviceMonitor = {
            enabled = true
          }
        }
      }
      
      server = {
        replicas = var.environment == "prod" ? 3 : 1
        service = {
          type = "LoadBalancer"
          annotations = {
            "service.beta.kubernetes.io/aws-load-balancer-type" = "nlb"
          }
        }
        metrics = {
          enabled = true
          serviceMonitor = {
            enabled = true
          }
        }
        ingress = {
          enabled = true
          ingressClassName = "nginx"
          annotations = {
            "nginx.ingress.kubernetes.io/ssl-redirect" = "true"
            "nginx.ingress.kubernetes.io/backend-protocol" = "GRPC"
          }
          hosts = [
            {
              host = "argocd.${var.domain_name}"
              paths = [
                {
                  path = "/"
                  pathType = "Prefix"
                }
              ]
            }
          ]
          tls = [
            {
              secretName = "argocd-server-tls"
              hosts = ["argocd.${var.domain_name}"]
            }
          ]
        }
      }
      
      repoServer = {
        replicas = var.environment == "prod" ? 3 : 1
        metrics = {
          enabled = true
          serviceMonitor = {
            enabled = true
          }
        }
      }
      
      redis = {
        enabled = true
        metrics = {
          enabled = true
          serviceMonitor = {
            enabled = true
          }
        }
      }
      
      # ApplicationSet controller
      applicationSet = {
        enabled = true
        replicas = var.environment == "prod" ? 2 : 1
      }
      
      # Notifications
      notifications = {
        enabled = true
      }
    })
  ]

  depends_on = [kubernetes_namespace.argocd]
}

# ArgoCD Application for the main app
resource "kubectl_manifest" "argocd_application" {
  yaml_body = yamlencode({
    apiVersion = "argoproj.io/v1alpha1"
    kind       = "Application"
    metadata = {
      name      = "${var.project_name}-${var.environment}"
      namespace = "argocd"
      finalizers = ["resources-finalizer.argocd.argoproj.io"]
    }
    spec = {
      project = "default"
      source = {
        repoURL        = var.git_repository_url
        targetRevision = var.git_branch
        path           = "k8s/overlays/${var.environment}"
        helm = {
          valueFiles = ["values-${var.aws_region}.yaml"]
        }
      }
      destination = {
        server    = "https://kubernetes.default.svc"
        namespace = "${var.project_name}-${var.environment}"
      }
      syncPolicy = {
        automated = {
          prune    = true
          selfHeal = true
        }
        syncOptions = [
          "CreateNamespace=true",
          "PrunePropagationPolicy=foreground"
        ]
      }
    }
  })

  depends_on = [helm_release.argocd]
}

# ArgoCD ApplicationSet for multi-region deployment
resource "kubectl_manifest" "argocd_applicationset" {
  yaml_body = yamlencode({
    apiVersion = "argoproj.io/v1alpha1"
    kind       = "ApplicationSet"
    metadata = {
      name      = "${var.project_name}-${var.environment}-multi-region"
      namespace = "argocd"
    }
    spec = {
      generators = [
        {
          list = {
            elements = [
              for region in var.regions : {
                region = region
                cluster = "https://kubernetes.default.svc"
              }
            ]
          }
        }
      ]
      template = {
        metadata = {
          name = "${var.project_name}-${var.environment}-{{region}}"
        }
        spec = {
          project = "default"
          source = {
            repoURL        = var.git_repository_url
            targetRevision = var.git_branch
            path           = "k8s/overlays/${var.environment}"
            helm = {
              valueFiles = ["values-{{region}}.yaml"]
              parameters = [
                {
                  name  = "region"
                  value = "{{region}}"
                }
              ]
            }
          }
          destination = {
            server    = "{{cluster}}"
            namespace = "${var.project_name}-${var.environment}"
          }
          syncPolicy = {
            automated = {
              prune    = true
              selfHeal = true
            }
            syncOptions = [
              "CreateNamespace=true",
              "PrunePropagationPolicy=foreground"
            ]
          }
        }
      }
    }
  })

  depends_on = [helm_release.argocd]
}

