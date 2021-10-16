"""Main module for hypha."""
from hypha.server import get_argparser, start_server

arg_parser = get_argparser()
opt = arg_parser.parse_args()
start_server(opt)
