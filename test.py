from app.models.mistral_model import generate_response

print(
    generate_response(
        "Solve: If a fair coin is tossed twice, what is the probability of getting exactly one head?"
    )
)