from connect_to_db import *
import argparse

connector = TheiaConnect(dbname='postgres', user='kishanp',
                         host='ec2-54-212-196-78.us-west-2.compute.amazonaws.com',  # noqa
                         port='5432', password='Theia628')

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--schema", required=True, help="name of schema")
args = vars(ap.parse_args())
schema = args['schema']

# students table
connector.create_table(schema, "students", "sid int, firstname varchar(255), \
                                lastname varchar(255)")
# class table
connector.create_table(schema, "class", "tid int, cid int, classname varchar(255)")  # noqa
# user table
connector.create_table(schema, "user", "tid int, firstname varchar(255),\
                                lastname varchar(255), username varchar(255), \
                                password varchar(255)")
# attendance table
connector.create_table(schema, "attendance", "sid int, cid int, dt timestamp")
# schedule table
connector.create_table(schema, "schedule", "sid int, cid int")
