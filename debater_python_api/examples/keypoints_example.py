from debater_python_api.api.clients.keypoints_client import KpsClient
from debater_python_api.api.debater_api import DebaterApi

debater_api = DebaterApi('PUT_YOUR_API_KEY_HERE')
keypoints_client = debater_api.get_keypoints_client()

comments_texts = [
    'Cannabis has detrimental effects on cognition and memory, some of which are irreversible.',
    'Cannabis can severely impact memory and productivity in its consumers.',
    'Cannabis harms the memory and learning capabilities of its consumers.',
    'Frequent use can impair cognitive ability.',
    'Cannabis harms memory, which in the long term hurts progress and can hurt people',
    'Frequent marijuana use can seriously affect short-term memory.',
    'Marijuana is very addictive, and therefore very dangerous'
    'Cannabis is addictive and very dangerous for use.',
    'Cannabis can be very harmful and addictive, especially for young people',
    'Cannabis is very addictive.'
                  ]

KpsClient.init_logger()
keypoint_matchings_result = keypoints_client.run_full_kps_flow(domain="keypoints_example_domain", comments_texts=comments_texts)
keypoint_matchings_result.print_result(n_sentences_per_kp=10, title="KPS Example")
