from alpaca.data.historical.corporate_actions import CorporateActionsClient
from alpaca.data.requests import CorporateActionsRequest
from alpaca.data.enums import CorporateActionsType
from datetime import date

API_KEY = "PKPLV2083K7HCYJGHY6U"
SECRET_KEY = "h6Jrvcpg3BBdADHlPhE8iX163GMfMufGO8zUGnrU"


# Initialize client
client = CorporateActionsClient(api_key=API_KEY, secret_key=SECRET_KEY)

# Prepare request params
request_params = CorporateActionsRequest(
    symbols=["TSLA"],                 # Symbols to get data for
    start=date(2022, 1, 1),
    end=date(2022, 12, 31),               # End date filter
    types=[CorporateActionsType.FORWARD_SPLIT]
)

# Get corporate actions
actions = client.get_corporate_actions(request_params)

# Print the data
if not actions:
    print("No forward splits found.")
else:
        print(actions)