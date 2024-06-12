import requests
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to fetch NFL odds data
def fetch_nfl_events(startsAfter, startsBefore):
    url = f'https://sportsgameodds.com/api/v1/events?leagueID=NFL&startsAfter={startsAfter}&startsBefore={startsBefore}'
    headers = {'x-api-key': <YOUR_API_KEY>}

    all_events = []
    next_cursor = None

    while True:
        try:
            response = requests.get(url, params={'cursor': next_cursor}, headers = headers)
            json = response.json()
            all_events.extend(json['data']])
            next_cursor = data.get('nextCursor')
            if not next_cursor:
                break
        except Exception as e:
            print(f'Error fetching NFL odds: {e}')
            break

    return all_events

# Function to preprocess data and extract features
def preprocess_data(events):
    features = []
    labels = []
    for event in events:
        # Extract relevant features (example: home team odds, away team odds)
        home_odds = event['odds']['points-home-game-ml-home']['odds']
        away_odds = event['odds']['points-away-game-ml-away']['odds']
        features.append([home_odds, away_odds])

        # Extract label (example: 1 if home team won, 0 otherwise)
        home_score = event['results']['game']['home']['points']
        away_score = event['results']['game']['away']['points']
        labels.append(1 if home_score > away_score else 0)

    return features, labels

# Function to train machine learning model
def train_model(features, labels):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train a RandomForestClassifier model (you can replace this with any other model)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model accuracy: {accuracy}')

    return model

# Function to make predictions using the trained model
def make_predictions(model, events):
    bets = []
    for event in events:
        home_odds = event['odds']['points-home-game-ml-home']['odds']
        away_odds = event['odds']['points-away-game-ml-away']['odds']

        prediction = model.predict([[home_odds, away_odds]])
        if prediction == 1:
            bets.append(f'Place a bet on {event["home_team"]} to win against {event["away_team"]}')
        else:
            bets.append(f'Place a bet on {event["away_team"]} to win against {event["home_team"]}')
    return bets

def main():
    today = datetime.today()
    next_week = today + timedelta(days=7)
    last_week = today - timedelta(days=7)

    # Fetch this weeks NFL odds data
    next_week_nfl_events = fetch_nfl_events(today, next_week)

    # Fetch last weeks NFL odds data and results
    last_week_nfl_events = fetch_nfl_events(last_week, today)

    if last_week_nfl_events:
        # Preprocess data
        features, labels = preprocess_data(last_week_nfl_events)

        # Train machine learning model
        model = train_model(features, labels)
    else:
        print('No previous NFL odds available.')

    if next_week_nfl_events:
        # Make predictions
        bets = make_predictions(model, next_week_nfl_events)
        for bet in bets:
            print(bet)
    else:
        print('No NFL odds available for the upcoming week.')

if __name__ == "__main__":
    main()
