# ─────────────────────────────────────────────
# outputs.tf — Terraform Outputs
# ─────────────────────────────────────────────

output "ec2_public_ip" {
  description = "Elastic IP of the EC2 instance"
  value       = aws_eip.spaceship_eip.public_ip
}

output "fastapi_url" {
  description = "FastAPI endpoint URL"
  value       = "http://${aws_eip.spaceship_eip.public_ip}:8000"
}

output "fastapi_docs_url" {
  description = "FastAPI Swagger UI URL"
  value       = "http://${aws_eip.spaceship_eip.public_ip}:8000/docs"
}

output "fastapi_health_url" {
  description = "FastAPI health check URL"
  value       = "http://${aws_eip.spaceship_eip.public_ip}:8000/health"
}

output "mlflow_url" {
  description = "MLflow UI URL"
  value       = "http://${aws_eip.spaceship_eip.public_ip}:5001"
}

output "ssh_command" {
  description = "SSH command to connect to EC2"
  value       = "ssh -i ${var.key_pair_name}.pem ubuntu@${aws_eip.spaceship_eip.public_ip}"
}
