import argparse
import slackweb

def main(config):
    slack = slackweb.Slack(url='https://hooks.slack.com/services/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')
    slack.notify(text=f"{config.message}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--message', type=str, default='message')
    
    config = parser.parse_args()
    
    main(config)