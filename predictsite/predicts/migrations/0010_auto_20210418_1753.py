# Generated by Django 3.1.7 on 2021-04-18 10:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predicts', '0009_auto_20210418_1746'),
    ]

    operations = [
        migrations.AlterField(
            model_name='predict',
            name='xray',
            field=models.ImageField(null=True, upload_to='xray/'),
        ),
    ]
