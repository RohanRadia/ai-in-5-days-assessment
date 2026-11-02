# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================================================
# Enterprise Cloud FinOps & Sustainability Concierge Agent - Main Terraform IaC
# Provisions:
# 1. Google Cloud APIs (Vertex AI, Cloud Run, Secret Manager, Cloud Logging,
#    Cloud Trace, Cloud DLP, Cloud SQL, BigQuery)
# 2. Secret Manager Secrets (zero hardcoded credentials)
# 3. Least-Privilege Agent Service Account & IAM Role Bindings
# 4. Cloud SQL PostgreSQL Instance for Persistent ADK Session State
# 5. Cloud Run v2 Service hosting the ADK FastAPI / A2A Container
# ============================================================================

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.30.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.6.0"
    }
  }
}

variable "project_id" {
  type        = string
  description = "Google Cloud Project ID for the FinOps & Sustainability Concierge Agent."
}

variable "region" {
  type        = string
  description = "Primary Google Cloud region for Cloud Run and Vertex AI workloads."
  default     = "us-central1"
}

variable "service_name" {
  type        = string
  description = "Name of the Cloud Run service and associated resources."
  default     = "finops-concierge"
}

variable "container_image" {
  type        = string
  description = "Container image URI deployed to Cloud Run."
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ----------------------------------------------------------------------------
# 1. Enable Required Google Cloud APIs (Vertex AI, Cloud Run, Secret Manager,
#    Cloud Logging, Cloud Trace, Cloud DLP, Cloud SQL)
# ----------------------------------------------------------------------------
locals {
  required_apis = [
    "aiplatform.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "logging.googleapis.com",
    "cloudtrace.googleapis.com",
    "telemetry.googleapis.com",
    "dlp.googleapis.com",
    "sqladmin.googleapis.com",
    "bigquery.googleapis.com",
    "cloudbuild.googleapis.com",
    "iam.googleapis.com",
  ]

  agent_iam_roles = [
    "roles/aiplatform.user",
    "roles/logging.logWriter",
    "roles/cloudtrace.agent",
    "roles/secretmanager.secretAccessor",
    "roles/dlp.user",
    "roles/cloudsql.client",
    "roles/bigquery.dataViewer",
    "roles/bigquery.jobUser",
  ]
}

resource "google_project_service" "enabled_apis" {
  for_each           = toset(local.required_apis)
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ----------------------------------------------------------------------------
# 2. Least-Privilege Service Account & IAM Role Bindings
# ----------------------------------------------------------------------------
resource "google_service_account" "finops_agent_sa" {
  account_id   = "${var.service_name}-sa"
  display_name = "Enterprise FinOps & Sustainability Concierge Service Account"
  project      = var.project_id
  depends_on   = [google_project_service.enabled_apis]
}

resource "google_project_iam_member" "finops_agent_iam" {
  for_each   = toset(local.agent_iam_roles)
  project    = var.project_id
  role       = each.value
  member     = "serviceAccount:${google_service_account.finops_agent_sa.email}"
  depends_on = [google_service_account.finops_agent_sa]
}

# ----------------------------------------------------------------------------
# 3. Google Cloud Secret Manager Secrets (Zero Hardcoded Credentials)
# ----------------------------------------------------------------------------
resource "random_password" "db_password" {
  length  = 24
  special = true
}

resource "google_secret_manager_secret" "session_db_password" {
  project   = var.project_id
  secret_id = "${var.service_name}-db-password"

  replication {
    auto {}
  }

  depends_on = [google_project_service.enabled_apis]
}

resource "google_secret_manager_secret_version" "session_db_password_v1" {
  secret      = google_secret_manager_secret.session_db_password.id
  secret_data = random_password.db_password.result
}

resource "google_secret_manager_secret" "finops_billing_export_token" {
  project   = var.project_id
  secret_id = "FINOPS_BILLING_EXPORT_TOKEN"

  replication {
    auto {}
  }

  depends_on = [google_project_service.enabled_apis]
}

# ----------------------------------------------------------------------------
# 4. Cloud SQL PostgreSQL Instance for Persistent ADK Session State
# ----------------------------------------------------------------------------
resource "google_sql_database_instance" "session_db" {
  project             = var.project_id
  name                = "${var.service_name}-session-db"
  database_version    = "POSTGRES_15"
  region              = var.region
  deletion_protection = false

  settings {
    tier = "db-custom-1-3840"

    database_flags {
      name  = "cloudsql.iam_authentication"
      value = "on"
    }
  }

  depends_on = [google_project_service.enabled_apis]
}

resource "google_sql_database" "adk_sessions" {
  project  = var.project_id
  name     = var.service_name
  instance = google_sql_database_instance.session_db.name
}

# ----------------------------------------------------------------------------
# 5. Cloud Run v2 Service for the FinOps & Sustainability Concierge Agent
# ----------------------------------------------------------------------------
resource "google_cloud_run_v2_service" "finops_concierge" {
  name                = var.service_name
  location            = var.region
  project             = var.project_id
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL"

  template {
    service_account                  = google_service_account.finops_agent_sa.email
    max_instance_request_concurrency = 16
    session_affinity                 = true

    scaling {
      min_instance_count = 1
      max_instance_count = 10
    }

    containers {
      image = var.container_image

      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }

      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "TRUE"
      }

      env {
        name  = "OTEL_SERVICE_NAME"
        value = var.service_name
      }

      env {
        name  = "INSTANCE_CONNECTION_NAME"
        value = google_sql_database_instance.session_db.connection_name
      }

      env {
        name  = "DB_NAME"
        value = google_sql_database.adk_sessions.name
      }

      env {
        name = "DB_PASS"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.session_db_password.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "FINOPS_BILLING_EXPORT_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.finops_billing_export_token.secret_id
            version = "latest"
          }
        }
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }
    }

    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [google_sql_database_instance.session_db.connection_name]
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_project_service.enabled_apis,
    google_project_iam_member.finops_agent_iam,
    google_secret_manager_secret_version.session_db_password_v1,
  ]
}

output "cloud_run_service_url" {
  description = "Deployed URL of the FinOps & Sustainability Concierge Cloud Run service."
  value       = google_cloud_run_v2_service.finops_concierge.uri
}
