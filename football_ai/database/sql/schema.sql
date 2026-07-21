CREATE TABLE IF NOT EXISTS teams (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(120) NOT NULL UNIQUE,
    fifa_code VARCHAR(8),
    country_code VARCHAR(8),
    confederation VARCHAR(32),
    current_elo DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_teams_name ON teams (name);
CREATE INDEX IF NOT EXISTS ix_teams_fifa_code ON teams (fifa_code);

CREATE TABLE IF NOT EXISTS players (
    id BIGSERIAL PRIMARY KEY,
    team_id BIGINT REFERENCES teams(id),
    external_id VARCHAR(64),
    full_name VARCHAR(120) NOT NULL,
    position VARCHAR(32),
    preferred_foot VARCHAR(16),
    market_value_eur DOUBLE PRECISION,
    date_of_birth DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_players_team_id ON players (team_id);
CREATE INDEX IF NOT EXISTS ix_players_full_name ON players (full_name);

CREATE TABLE IF NOT EXISTS matches (
    id BIGSERIAL PRIMARY KEY,
    external_id VARCHAR(64) NOT NULL,
    source VARCHAR(32) NOT NULL,
    -- Kept generic on purpose: this platform is meant to grow into a
    -- universal (international + club) prediction engine. V1 only ingests
    -- international competitions, but every match is tagged from day one so
    -- club football can be added later without a backfill migration.
    competition_type VARCHAR(16) NOT NULL DEFAULT 'international'
        CONSTRAINT ck_matches_competition_type CHECK (competition_type IN ('international', 'club')),
    competition_id VARCHAR(64),
    competition_name VARCHAR(120),
    season_name VARCHAR(64),
    tournament VARCHAR(120),
    status VARCHAR(32),
    match_date TIMESTAMPTZ NOT NULL,
    stadium_name VARCHAR(120),
    city VARCHAR(120),
    country VARCHAR(120),
    venue_latitude DOUBLE PRECISION,
    venue_longitude DOUBLE PRECISION,
    neutral_venue BOOLEAN NOT NULL DEFAULT FALSE,
    referee VARCHAR(120),
    attendance INTEGER,
    home_team_id BIGINT NOT NULL REFERENCES teams(id),
    away_team_id BIGINT NOT NULL REFERENCES teams(id),
    home_score INTEGER,
    away_score INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_matches_source_external_id UNIQUE (source, external_id)
);

CREATE INDEX IF NOT EXISTS ix_matches_date ON matches (match_date);
CREATE INDEX IF NOT EXISTS ix_matches_date_teams
    ON matches (match_date, home_team_id, away_team_id);
CREATE INDEX IF NOT EXISTS ix_matches_competition_type ON matches (competition_type);

CREATE TABLE IF NOT EXISTS lineups (
    id BIGSERIAL PRIMARY KEY,
    match_id BIGINT NOT NULL REFERENCES matches(id),
    team_id BIGINT NOT NULL REFERENCES teams(id),
    player_id BIGINT NOT NULL REFERENCES players(id),
    started BOOLEAN NOT NULL DEFAULT TRUE,
    minutes_played INTEGER,
    shirt_number INTEGER,
    position VARCHAR(32),
    is_captain BOOLEAN NOT NULL DEFAULT FALSE,
    age DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_lineups_match_player UNIQUE (match_id, team_id, player_id)
);

CREATE TABLE IF NOT EXISTS team_statistics (
    id BIGSERIAL PRIMARY KEY,
    match_id BIGINT NOT NULL REFERENCES matches(id),
    team_id BIGINT NOT NULL REFERENCES teams(id),
    is_home BOOLEAN NOT NULL,
    goals INTEGER,
    xg DOUBLE PRECISION,
    xga DOUBLE PRECISION,
    possession_pct DOUBLE PRECISION,
    shots INTEGER,
    shots_on_target INTEGER,
    corners INTEGER,
    yellow_cards INTEGER,
    red_cards INTEGER,
    fouls INTEGER,
    offsides INTEGER,
    passes_completed INTEGER,
    passes_attempted INTEGER,
    pressing_intensity DOUBLE PRECISION,
    duels_won INTEGER,
    saves INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_team_statistics_match_team UNIQUE (match_id, team_id)
);

CREATE INDEX IF NOT EXISTS ix_team_statistics_match_home
    ON team_statistics (match_id, is_home);

CREATE TABLE IF NOT EXISTS player_statistics (
    id BIGSERIAL PRIMARY KEY,
    match_id BIGINT NOT NULL REFERENCES matches(id),
    team_id BIGINT NOT NULL REFERENCES teams(id),
    player_id BIGINT NOT NULL REFERENCES players(id),
    minutes_played INTEGER,
    goals INTEGER,
    assists INTEGER,
    xg DOUBLE PRECISION,
    xa DOUBLE PRECISION,
    shots INTEGER,
    shots_on_target INTEGER,
    touches INTEGER,
    passes_completed INTEGER,
    key_passes INTEGER,
    tackles INTEGER,
    interceptions INTEGER,
    saves INTEGER,
    yellow_cards INTEGER,
    red_cards INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_player_statistics_match_player UNIQUE (match_id, team_id, player_id)
);

CREATE TABLE IF NOT EXISTS elo_ratings (
    id BIGSERIAL PRIMARY KEY,
    team_id BIGINT NOT NULL REFERENCES teams(id),
    rating_date DATE NOT NULL,
    rating DOUBLE PRECISION NOT NULL,
    rank INTEGER,
    source VARCHAR(32) NOT NULL DEFAULT 'world_football_elo',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_elo_team_date_source UNIQUE (team_id, rating_date, source)
);

CREATE INDEX IF NOT EXISTS ix_elo_ratings_team_date
    ON elo_ratings (team_id, rating_date DESC);

CREATE TABLE IF NOT EXISTS injuries (
    id BIGSERIAL PRIMARY KEY,
    team_id BIGINT NOT NULL REFERENCES teams(id),
    player_id BIGINT REFERENCES players(id),
    injury_date DATE NOT NULL,
    return_date DATE,
    severity VARCHAR(32),
    description VARCHAR(255),
    source VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_injuries_team_player
    ON injuries (team_id, player_id, injury_date DESC);

CREATE TABLE IF NOT EXISTS weather (
    id BIGSERIAL PRIMARY KEY,
    match_id BIGINT NOT NULL REFERENCES matches(id),
    temperature_c DOUBLE PRECISION,
    apparent_temperature_c DOUBLE PRECISION,
    precipitation_mm DOUBLE PRECISION,
    wind_speed_kph DOUBLE PRECISION,
    humidity_pct DOUBLE PRECISION,
    pressure_hpa DOUBLE PRECISION,
    cloud_cover_pct DOUBLE PRECISION,
    weather_code INTEGER,
    source VARCHAR(32) NOT NULL DEFAULT 'open_meteo',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_weather_match UNIQUE (match_id)
);

CREATE TABLE IF NOT EXISTS model_versions (
    id BIGSERIAL PRIMARY KEY,
    version VARCHAR(64) NOT NULL,
    model_name VARCHAR(64) NOT NULL,
    target_name VARCHAR(64) NOT NULL,
    task_type VARCHAR(32) NOT NULL,
    metrics JSONB,
    feature_names JSONB,
    artifact_path VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_model_version_name UNIQUE (version, model_name, target_name)
);

CREATE INDEX IF NOT EXISTS ix_model_versions_active
    ON model_versions (is_active, target_name);

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    match_id BIGINT REFERENCES matches(id),
    model_version_id BIGINT NOT NULL REFERENCES model_versions(id),
    request_payload JSONB NOT NULL,
    outcome_probabilities JSONB,
    predicted_scores JSONB,
    expected_statistics JSONB,
    raw_prediction JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_predictions_match_model
    ON predictions (match_id, model_version_id);
