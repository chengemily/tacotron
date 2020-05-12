import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


sentences = [
  # From July 8, 2017 New York Times:
  'Scientists at the CERN laboratory say they have discovered a new particle.',
  'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
  'President Trump met with other leaders at the Group of 20 conference.',
  'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
  # From Google's Tacotron example page:
  'Generative adversarial network or variational auto-encoder.',
  'The buses aren\'t the problem, they actually provide a solution.',
  'Does the quick brown fox jump over the lazy dog?',
  'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
  # More, based on https://www.researchgate.net/figure/Distribution-of-English-sentences-according-to-their-lengths-in-words_fig4_221628889
  'You\'re still not completely convinced that that\'s going to happen, are you?',
  'Tom thinks that it\'s time for Mary to ask for help.',
  'I think that he is the only vegetarian here.',
  'Sami was a zombie, and he didn\'t know what was going on.',
  'I thought you said I have to do that.',
  'In silence, she placed her left hand in my right hand.',
  'Both my parents are musicians.',
  'Did you guys talk about me while I was gone?',
  'Just tell me what happens.',
  'I can\'t sit around waiting any longer.',
  'Are you still interested in the job?',
  'Jesus wept.'
]


def get_output_base_path(checkpoint_path, model_name):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = '%s-eval-%d' % (model_name, int(m.group(1)) if m else 'eval')
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  synth = Synthesizer()
  synth.load(args.checkpoint)
  base_path = get_output_base_path(args.checkpoint, args.model_name)
  for i, text in enumerate(sentences):
    path = '%s-%d.wav' % (base_path, i)
    print('Synthesizing: %s' % path)
    with open(path, 'wb') as f:
      f.write(synth.synthesize(text))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--model_name', required=True)
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()