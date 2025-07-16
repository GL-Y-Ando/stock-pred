# Python Development Environment

A clean, containerized Python development environment with essential data science and web scraping libraries.

## Quick Start

1. Clone this repository
2. Run the following command:
   ```bash
   docker compose up
   ```
3. Access the container:
   ```bash
   docker exec -it python-dev-container python
   ```

## What's Included

- **Python 3.11** - Latest stable Python version
- **Core Libraries:**
  - `numpy` - Numerical computing
  - `pandas` - Data manipulation and analysis
  - `requests` - HTTP library for API calls
  - `jupyter` - Interactive notebooks
  - `matplotlib` & `seaborn` - Data visualization
  - `python-dotenv` - Environment variable management

## Project Structure

```
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── README.md
├── src/              # Your Python source code
├── data/             # Data files
└── notebooks/        # Jupyter notebooks
```

## Usage

### Interactive Python Shell
```bash
docker exec -it python-dev-container python
```

### Run Python Scripts
```bash
docker exec -it python-dev-container python src/your_script.py
```

### Jupyter Notebook (Optional)
```bash
docker exec -it python-dev-container jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
Then access http://localhost:8888 in your browser.

### Install Additional Packages
```bash
docker exec -it python-dev-container pip install package_name
```

## Development Tips

- All your code should go in the `src/` directory
- Data files can be stored in the `data/` directory
- The container has persistent volumes, so your files won't be lost
- The container stays running until you stop it with `docker compose down`

## Customization

To add more Python packages:
1. Add them to `requirements.txt`
2. Rebuild the container: `docker compose up --build`