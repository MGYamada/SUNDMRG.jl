# Agent Instructions

## Local Network Configuration

- Do not record the user's local network configuration or connection details in
  repository files, documentation, source comments, test fixtures, generated
  artifacts, saved diagnostic logs, commit messages, issue/PR descriptions,
  or persistent agent notes.
- This includes private IP addresses, subnet/gateway/DNS settings, local
  hostnames, SSIDs, router settings, port forwarding and firewall rules,
  and SSH endpoints, usernames, key paths, and credentials.
- Use connection details only as needed for the current authorized task.
  Use placeholders such as `<GPU_HOST>` and `<SSH_USER>` in saved examples,
  launch commands, and instructions.
- Redact network and connection details before saving diagnostic output or
  publishing validation reports.
- GPU/MPI validation records may include GPU models/counts, OS and software
  versions, numerical results, and commands with placeholders. Omit local
  network setup and remote-access configuration.
