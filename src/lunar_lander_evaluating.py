

def evalute_model(episodes, env, model, episode_rewards, episode_lengths, success_threshold):
    for episode in range():
            observation, info = env.reset()
            done = False
            score = 0
            length = 0
            while not done:
                env.render()
                action, _ = model.predict(observation)
                try:
                    observation, reward, done, info = env.step(action)
                except ValueError:
                    observation, reward, done, truncated, info = env.step(action)
                    done = done or truncated
                score += reward
                length += 1
            
            episode_rewards.append(score)
            episode_lengths.append(length)
            if score >= success_threshold:
                successes += 1

            print(f"Episode {episode + 1} finished with score: {score}")