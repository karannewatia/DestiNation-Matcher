# Import flask deps
from flask import request, render_template, \
	flash, g, session, redirect, url_for, jsonify, abort

# For decorators around routes
from functools import wraps 

# Import for pass / encryption 
from werkzeug import check_password_hash, generate_password_hash 

# Marshmallow 
from marshmallow import ValidationError

# Import socketio for socket creation in this module 
from app import socketio

# IMPORT THE BLUEPRINT APP OBJECT 
from app.irsystem import irsystem 
